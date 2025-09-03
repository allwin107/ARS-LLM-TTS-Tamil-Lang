# Tamil Real-Time Conversation System

A real-time conversational system that supports Tamil language processing through speech recognition, AI-powered conversation, and text-to-speech synthesis.

## Features

- Real-time speech recognition using Faster Whisper
- Voice Activity Detection (VAD) for improved speech segmentation
- Conversation capabilities using Google's Gemini model
- Text-to-speech synthesis using Edge TTS
- Support for Tamil language (with voice: ta-IN-PallaviNeural)
- Both CLI and Gradio web interface options

## Prerequisites

- Python 3.11 or higher
- Virtual environment (recommended)
- Required models will be downloaded automatically:
  - Faster Whisper models (tiny, small, medium, large-v2, base)
  - Edge TTS voice models
  - Google Gemini API access

## Installation

1. Clone the repository:
```bash
git clone https://github.com/allwin107/ARS-LLM-TTS.git
cd tamil-rt
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # For Windows PowerShell
```

3. Install the required dependencies:
```bash
pip install gradio av edge-tts google-generativeai python-dotenv numpy huggingface_hub
```

4. Download and extract the models:
```bash
# Download models.zip from: [Add your preferred hosting link here]
# Then extract it:
Expand-Archive -Path models.zip -DestinationPath . -Force  # For Windows PowerShell
# OR
unzip models.zip  # For Linux/Mac
```

Note: Due to file size limitations, the models.zip file is hosted externally. You can also download individual models using the Hugging Face Hub as described in the Model Management section.

5. Set up environment variables:
Create a `.env` file in the project root with the following variables:
```env
GOOGLE_API_KEY=your_gemini_api_key
TTS_VOICE=ta-IN-PallaviNeural
GEMINI_MODEL=gemini-1.5-flash
```

## Usage

### Command Line Interface
Run the main script for CLI interaction:
```bash
python main.py
```

### Web Interface
Run the Gradio web interface:
```bash
python app_gradio.py
```

The web interface provides an intuitive way to interact with the system, featuring:
- Microphone input for speech recognition
- Real-time transcription display
- AI-generated responses
- Text-to-speech playback

## Model Management

The system uses several AI models that can be managed through the web interface:

### ASR Models
The repository includes pre-downloaded Faster-Whisper models in the `models/faster-whisper` directory. You'll have access to all model sizes without needing to download them separately.

In the web UI Live tab you can:
- **Preload Selected**: Use a specific Faster-Whisper model size (tiny/base/small/medium/large-v2)
- **Preload All**: Load all supported model sizes at once
- **Load ASR**: Switch between models instantly

Models are located in `./models/faster-whisper/<size>`. The system will use these local files instead of downloading them again.

### Model Selection Guide

- **tiny**: Fastest, lowest accuracy
- **base**: Good balance of speed and accuracy
- **small**: Better accuracy
- **medium**: High accuracy
- **large-v2**: Highest accuracy

## System Architecture

- Speech recognition processes audio in real-time at 16kHz
- Voice Activity Detection uses 20ms frame size
- Playback operates at 24kHz for optimal quality
- System requirements depend on the chosen Whisper model size

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your chosen license here]

## Acknowledgments

- Faster Whisper for efficient speech recognition
- Google Gemini for natural language processing
- Microsoft Edge TTS for high-quality speech synthesis
- WebRTC VAD for voice activity detection
