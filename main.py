import asyncio
import io
import os
import queue
import re
import threading
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad
from dotenv import load_dotenv
from faster_whisper import WhisperModel
import google.generativeai as genai
import edge_tts
import av


# ---------------------------
# Configuration
# ---------------------------
ASR_SAMPLE_RATE = 16000  # Hz for VAD+ASR input
ASR_FRAME_MS = 20        # 10/20/30 allowed by VAD
ASR_FRAME_SAMPLES = int(ASR_SAMPLE_RATE * ASR_FRAME_MS / 1000)
ASR_DTYPE = np.int16
VAD_AGGRESSIVENESS = 2
LANG_CODE = "ta"  # Tamil

TTS_VOICE = os.getenv("TTS_VOICE", "ta-IN-PallaviNeural")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

PLAYBACK_RATE = 24000  # We will decode TTS to 24 kHz mono PCM
PLAYBACK_BLOCK = 2048

MAX_TOKENS_PER_PHRASE = 3
PUNCT_BOUNDARY_RE = re.compile(r"[.!?…]")


@dataclass
class AudioSegment:
    samples: np.ndarray  # int16 mono at 16 kHz


class MicrophoneStreamer:
    def __init__(self, sample_rate: int, frame_samples: int, dtype):
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.dtype = dtype
        self._q = queue.Queue()
        self._stop = threading.Event()
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            # Drop or log
            pass
        # Downmix if needed, ensure int16
        if indata.ndim == 2:
            mono = np.mean(indata, axis=1)
        else:
            mono = indata[:, 0] if indata.ndim == 2 else indata
        # Convert to int16 if not already
        if mono.dtype != ASR_DTYPE:
            # Assume float32 from device; scale
            if np.issubdtype(mono.dtype, np.floating):
                mono = np.clip(mono, -1.0, 1.0)
                mono = (mono * 32767.0).astype(ASR_DTYPE)
            else:
                mono = mono.astype(ASR_DTYPE)
        # Split into exact VAD-sized frames
        for start in range(0, len(mono), self.frame_samples):
            frame = mono[start:start + self.frame_samples]
            if len(frame) == self.frame_samples:
                self._q.put(bytes(frame.tobytes()))

    def start(self):
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=self.frame_samples,
            callback=self._callback,
        )
        self._stream.start()

    def read(self, timeout: float = 1.0) -> bytes | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class VadSegmenter:
    def __init__(self, sample_rate: int, max_silence_ms: int = 400, min_voiced_ms: int = 200):
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.max_silence_frames = max_silence_ms // ASR_FRAME_MS
        self.min_voiced_frames = max(1, min_voiced_ms // ASR_FRAME_MS)
        self.reset()

    def reset(self):
        self.buffer = []
        self.voiced_frames = 0
        self.silence_frames = 0
        self.in_voiced = False

    def process_frame(self, frame_bytes: bytes) -> AudioSegment | None:
        is_speech = False
        try:
            is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
        except Exception:
            # Treat undecidable as non-speech
            is_speech = False

        self.buffer.append(frame_bytes)

        if is_speech:
            self.voiced_frames += 1
            self.silence_frames = 0
            self.in_voiced = True
        else:
            if self.in_voiced:
                self.silence_frames += 1
            # else ignore initial silence without flipping state

        # End segment if in voiced and enough trailing silence
        if self.in_voiced and self.silence_frames >= self.max_silence_frames and self.voiced_frames >= self.min_voiced_frames:
            pcm = b"".join(self.buffer)
            samples = np.frombuffer(pcm, dtype=ASR_DTYPE)
            seg = AudioSegment(samples=samples)
            self.reset()
            return seg
        return None


class PlaybackStreamer:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self._q = queue.Queue(maxsize=32)
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=PLAYBACK_BLOCK,
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, outdata, frames, time_info, status):
        if status:
            pass
        remaining = frames
        ptr = 0
        outdata.fill(0)
        while remaining > 0:
            try:
                chunk = self._q.get_nowait()
            except queue.Empty:
                break
            n = min(remaining, len(chunk))
            outdata[:n, 0] = chunk[:n]
            remaining -= n
            ptr += n
            if n < len(chunk):
                # put back remainder
                self._q.queue.appendleft(chunk[n:])
                break

    def enqueue(self, audio_f32: np.ndarray):
        if audio_f32.ndim > 1:
            audio_f32 = audio_f32[:, 0]
        # Split into manageable blocks
        start = 0
        while start < len(audio_f32):
            end = min(len(audio_f32), start + PLAYBACK_BLOCK)
            self._q.put(audio_f32[start:end].astype(np.float32))
            start = end


def decode_mp3_to_float32(mp3_bytes: bytes, dst_rate: int) -> np.ndarray:
    with av.open(io.BytesIO(mp3_bytes), format="mp3") as container:
        stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=dst_rate)
        frames = []
        for packet in container.demux(stream):
            for frame in packet.decode():
                frame_mono = frame
                if frame_mono.layout.name != "mono":
                    frame_mono = frame_mono.to_mono()
                frame_res = resampler.resample(frame_mono)
                if frame_res is None:
                    continue
                resampled_frames = frame_res if isinstance(frame_res, list) else [frame_res]
                for fr in resampled_frames:
                    arr = fr.to_ndarray()
                    # arr shape (channels, samples) -> (samples,)
                    if arr.ndim == 2:
                        arr = arr[0]
                    fmt_name = getattr(fr.format, "name", "")
                    if fmt_name.startswith("s16"):
                        arr = arr.astype(np.float32) / 32768.0
                    else:
                        arr = arr.astype(np.float32)
                    frames.append(arr)
        if frames:
            return np.concatenate(frames)
        return np.zeros(0, dtype=np.float32)


async def tts_phrase_to_playback(phrase: str, voice: str, playback: PlaybackStreamer):
    print(f"[TTS] Synthesizing phrase: {phrase}")
    # Stream TTS as MP3 chunks, then decode and enqueue
    communicate = edge_tts.Communicate(phrase, voice)
    buf = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf += chunk["data"]
    if len(buf) > 0:
        audio = decode_mp3_to_float32(bytes(buf), dst_rate=PLAYBACK_RATE)
        playback.enqueue(audio)
        print(f"[TTS] Enqueued audio, {len(audio)} samples @ {PLAYBACK_RATE} Hz")


def transcribe_segment(whisper: WhisperModel, seg: AudioSegment) -> str:
    # Convert int16 to float32 in [-1, 1]
    audio = seg.samples.astype(np.float32) / 32768.0
    segments, _ = whisper.transcribe(
        audio,
        language=LANG_CODE,
        vad_filter=False,
        beam_size=2,
        condition_on_previous_text=False,
    )
    text_parts = [s.text for s in segments]
    return (" ".join(text_parts)).strip()


def llm_stream_and_tts(gen_model, user_text: str, playback: PlaybackStreamer, loop: asyncio.AbstractEventLoop):
    sys_prompt = "You are a helpful assistant. Reply only in Tamil (ta). Keep responses brief and natural. Avoid transliteration."
    print("[LLM] Requesting stream …")
    response = gen_model.generate_content([
        {"text": sys_prompt},
        {"text": user_text}
    ], stream=True)

    llm_buf = ""
    tokens_since_flush = 0

    for chunk in response:
        piece = getattr(chunk, "text", None)
        if not piece:
            continue
        print(f"[LLM] chunk {len(piece)} chars")
        llm_buf += piece
        tokens_since_flush += len(piece.split())

        flushed_any = True
        while flushed_any:
            flushed_any = False
            m = PUNCT_BOUNDARY_RE.search(llm_buf)
            if m:
                idx = m.end()
                phrase = llm_buf[:idx].strip()
                llm_buf = llm_buf[idx:].strip()
                tokens_since_flush = 0
                fut = asyncio.run_coroutine_threadsafe(tts_phrase_to_playback(phrase, TTS_VOICE, playback), loop)
                fut.result()
                flushed_any = True

        if (tokens_since_flush >= MAX_TOKENS_PER_PHRASE or len(llm_buf) >= 12) and llm_buf.strip():
            phrase = llm_buf.strip()
            llm_buf = ""
            tokens_since_flush = 0
            fut = asyncio.run_coroutine_threadsafe(tts_phrase_to_playback(phrase, TTS_VOICE, playback), loop)
            fut.result()

    if llm_buf.strip():
        fut = asyncio.run_coroutine_threadsafe(tts_phrase_to_playback(llm_buf.strip(), TTS_VOICE, playback), loop)
        fut.result()


def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] Please set GEMINI_API_KEY in .env")
        return
    genai.configure(api_key=api_key)
    gen_model = genai.GenerativeModel(GEMINI_MODEL)

    print("[INIT] Loading Whisper model (small) for CPU …")
    device = "cpu"
    whisper = WhisperModel("small", device=device, compute_type="int8")

    print("[AUDIO] Starting microphone …")
    mic = MicrophoneStreamer(ASR_SAMPLE_RATE, ASR_FRAME_SAMPLES, ASR_DTYPE)
    mic.start()
    segmenter = VadSegmenter(ASR_SAMPLE_RATE)
    playback = PlaybackStreamer(PLAYBACK_RATE)

    print("[READY] Speak Tamil. Ctrl+C to stop.")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        while True:
            frame = mic.read(timeout=1.0)
            if frame is None:
                continue
            seg = segmenter.process_frame(frame)
            if seg is not None:
                print("[ASR] Transcribing segment …")
                text = transcribe_segment(whisper, seg)
                if not text:
                    continue
                print(f"[ASR] Text: {text}")
                # Stream LLM tokens and TTS phrases synchronously, using the loop for TTS
                llm_stream_and_tts(gen_model, text, playback, loop)
    except KeyboardInterrupt:
        print("\n[EXIT] Stopping …")
    finally:
        mic.stop()


if __name__ == "__main__":
    main()


