import asyncio
import io
import os
import threading

import numpy as np
import av
import edge_tts
import gradio as gr
from dotenv import load_dotenv
import google.generativeai as genai
from faster_whisper import WhisperModel
from main import (
    MicrophoneStreamer,
    VadSegmenter,
    PlaybackStreamer,
    AudioSegment,
    llm_stream_and_tts,
    ASR_SAMPLE_RATE,
    ASR_FRAME_SAMPLES,
    ASR_DTYPE,
)
from huggingface_hub import snapshot_download


# ---------------------------
# Helpers
# ---------------------------
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


# ---------------------------
# App setup
# ---------------------------
PLAYBACK_RATE = 24000
AVAILABLE_WHISPER_SIZES = ["tiny", "base", "small", "medium", "large-v2"]

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
voice_name = os.getenv("TTS_VOICE", "ta-IN-PallaviNeural")
whisper_model_name = os.getenv("WHISPER_MODEL", "small")
WHISPER_LOCAL_DIR = os.getenv("WHISPER_LOCAL_DIR", os.path.join(os.getcwd(), "models", "faster-whisper"))

if api_key:
    genai.configure(api_key=api_key)
    gen_model = genai.GenerativeModel(model_name)
else:
    gen_model = None

# ASR model (CPU by default)
whisper_device = "cpu"


def _repo_for_size(size: str) -> str:
    return f"Systran/faster-whisper-{size}"


def ensure_local_whisper(size: str) -> str:
    os.makedirs(os.path.join(WHISPER_LOCAL_DIR, size), exist_ok=True)
    repo_id = _repo_for_size(size)
    # This is idempotent; if already present, it reuses local files.
    local_path = snapshot_download(
        repo_id=repo_id,
        local_dir=os.path.join(WHISPER_LOCAL_DIR, size),
        local_dir_use_symlinks=False,
    )
    return local_path


def load_whisper_locally(size: str) -> WhisperModel:
    try:
        local_path = ensure_local_whisper(size)
        return WhisperModel(local_path, device=whisper_device, compute_type="int8")
    except Exception:
        # Fallback to remote name (will download if not cached by HF cache)
        return WhisperModel(size, device=whisper_device, compute_type="int8")


whisper = load_whisper_locally(whisper_model_name)
live_stop = threading.Event()


async def synthesize_tts_to_numpy(text: str, voice: str, dst_rate: int) -> tuple[int, np.ndarray]:
    communicate = edge_tts.Communicate(text, voice)
    buf = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf += chunk["data"]
    if len(buf) == 0:
        return dst_rate, np.zeros(0, dtype=np.float32)
    audio = decode_mp3_to_float32(bytes(buf), dst_rate=dst_rate)
    return dst_rate, audio


def handle_prompt(user_text: str) -> tuple[str, tuple[int, np.ndarray] | None]:
    if not user_text or not user_text.strip():
        return "", None
    if gen_model is None:
        return "[ERROR] Please set GEMINI_API_KEY in .env and restart.", None

    sys_prompt = "You are a helpful assistant. Reply only in Tamil (ta). Keep responses brief and natural. Avoid transliteration."
    try:
        response = gen_model.generate_content([
            {"text": sys_prompt},
            {"text": user_text.strip()},
        ])
        reply_text = (getattr(response, "text", None) or "").strip()
    except Exception as e:
        return f"[ERROR] LLM request failed: {e}", None

    if not reply_text:
        return "", None

    try:
        sr, audio = asyncio.run(synthesize_tts_to_numpy(reply_text, voice_name, PLAYBACK_RATE))
    except Exception as e:
        return reply_text, None

    return reply_text, (sr, audio)


def resample_to_16k(audio: np.ndarray, src_rate: int) -> np.ndarray:
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if src_rate == 16000:
        return audio.astype(np.float32)
    n_samples = int(len(audio) * 16000 / max(1, src_rate))
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_samples, endpoint=False)
    y_new = np.interp(x_new, x_old, audio.astype(np.float32))
    return y_new.astype(np.float32)


def transcribe_numpy_audio(audio_in: tuple[int, np.ndarray]) -> str:
    if audio_in is None:
        return ""
    sr, audio = audio_in
    if audio is None or len(audio) == 0:
        return ""
    audio_16k = resample_to_16k(audio, sr)
    segments, _ = whisper.transcribe(
        audio_16k,
        language="ta",
        vad_filter=False,
        beam_size=2,
        condition_on_previous_text=False,
    )
    text_parts = [s.text for s in segments]
    return (" ".join(text_parts)).strip()


def transcribe_segment_live(seg: AudioSegment) -> str:
    audio = seg.samples.astype(np.float32) / 32768.0
    segments, _ = whisper.transcribe(
        audio,
        language="ta",
        vad_filter=True,
        beam_size=5,
        condition_on_previous_text=False,
    )
    text_parts = [s.text for s in segments]
    return (" ".join(text_parts)).strip()


def _run_event_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


def start_live(_: None):
    live_stop.clear()
    mic = MicrophoneStreamer(ASR_SAMPLE_RATE, ASR_FRAME_SAMPLES, ASR_DTYPE)
    mic.start()
    segmenter = VadSegmenter(ASR_SAMPLE_RATE)
    playback = PlaybackStreamer(PLAYBACK_RATE)

    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=_run_event_loop, args=(loop,), daemon=True)
    loop_thread.start()

    transcript_accum = ""
    try:
        while not live_stop.is_set():
            frame = mic.read(timeout=1.0)
            if frame is None:
                yield transcript_accum, "Listening…"
                continue
            seg = segmenter.process_frame(frame)
            if seg is not None:
                text = transcribe_segment_live(seg)
                if not text:
                    yield transcript_accum, "Listening…"
                    continue
                transcript_accum = (transcript_accum + "\n" + text).strip()
                yield transcript_accum, "Speaking…"
                # Stream reply audio to speakers
                llm_stream_and_tts(gen_model, text, playback, loop)
                yield transcript_accum, "Listening…"
    finally:
        mic.stop()
        try:
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join(timeout=1.0)
            loop.close()
        except Exception:
            pass


def stop_live():
    live_stop.set()
    return "Stopped"


def handle_voice(audio_in: tuple[int, np.ndarray] | None) -> tuple[str, str, tuple[int, np.ndarray] | None]:
    transcript = transcribe_numpy_audio(audio_in)
    if not transcript:
        return "", "", None
    if gen_model is None:
        return transcript, "[ERROR] Please set GEMINI_API_KEY in .env and restart.", None
    sys_prompt = "You are a helpful assistant. Reply only in Tamil (ta). Keep responses brief and natural. Avoid transliteration."
    try:
        response = gen_model.generate_content([
            {"text": sys_prompt},
            {"text": transcript},
        ])
        reply_text = (getattr(response, "text", None) or "").strip()
    except Exception as e:
        return transcript, f"[ERROR] LLM request failed: {e}", None
    if not reply_text:
        return transcript, "", None
    try:
        sr, audio = asyncio.run(synthesize_tts_to_numpy(reply_text, voice_name, PLAYBACK_RATE))
    except Exception:
        return transcript, reply_text, None
    return transcript, reply_text, (sr, audio)


with gr.Blocks(title="Tamil RT - Web UI") as demo:
    gr.Markdown("**Tamil RT Web** — Text or voice in, voice out.")
    with gr.Tab("Text"):
        inp_text = gr.Textbox(label="Tamil input", placeholder="தமிழில் எழுதவும்…", lines=3)
        btn_text = gr.Button("Reply")
        out_text = gr.Textbox(label="Reply (Tamil)")
        out_audio = gr.Audio(label="TTS", type="numpy")
        btn_text.click(fn=handle_prompt, inputs=inp_text, outputs=[out_text, out_audio])
    with gr.Tab("Voice"):
        inp_audio = gr.Audio(sources=["microphone"], type="numpy")
        btn_voice = gr.Button("Transcribe and Reply")
        out_transcript = gr.Textbox(label="Transcript (Tamil)")
        out_reply = gr.Textbox(label="Reply (Tamil)")
        out_voice = gr.Audio(label="Reply TTS", type="numpy")
        btn_voice.click(fn=handle_voice, inputs=inp_audio, outputs=[out_transcript, out_reply, out_voice])
    with gr.Tab("Live (Local Mic)"):
        gr.Markdown("Real-time local mic conversation. Improves accuracy with stronger ASR.")
        info_row = gr.Row()
        with info_row:
            asr_model_dd = gr.Dropdown(AVAILABLE_WHISPER_SIZES, value=whisper_model_name, label="ASR model")
            reload_asr_btn = gr.Button("Load ASR")
            preload_sel_btn = gr.Button("Preload Selected")
            preload_all_btn = gr.Button("Preload All")
        live_row = gr.Row()
        with live_row:
            start_btn = gr.Button("Start Live")
            stop_btn = gr.Button("Stop")
        live_transcript = gr.Textbox(label="Heard (Tamil)", lines=6)
        live_status = gr.Textbox(label="Status")

        def load_asr(model_name: str):
            global whisper
            whisper = load_whisper_locally(model_name)
            return f"ASR loaded locally: {model_name}"

        def preload_selected(model_name: str):
            try:
                path = ensure_local_whisper(model_name)
                return f"Preloaded {model_name} at {path}"
            except Exception as e:
                return f"Preload failed for {model_name}: {e}"

        def preload_all():
            msgs = []
            for size in AVAILABLE_WHISPER_SIZES:
                try:
                    path = ensure_local_whisper(size)
                    msgs.append(f"{size}: ready at {path}")
                except Exception as e:
                    msgs.append(f"{size}: failed ({e})")
            return "\n".join(msgs)

        reload_asr_btn.click(load_asr, inputs=asr_model_dd, outputs=live_status)
        preload_sel_btn.click(preload_selected, inputs=asr_model_dd, outputs=live_status)
        preload_all_btn.click(preload_all, inputs=None, outputs=live_status)
        start_btn.click(start_live, inputs=None, outputs=[live_transcript, live_status])
        stop_btn.click(stop_live, inputs=None, outputs=live_status)


if __name__ == "__main__":
    demo.launch()


