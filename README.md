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
git clone https://github.com/allwin107/ARS-LLM-TTS-Tamil-Lang.git
cd ARS-LLM-TTS-Tamil-Lang
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # For Windows PowerShell
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Models Setup

You can set up Faster-Whisper models in multiple ways. Pick the method that suits your environment :

- Direct download (Google Drive):
  - Download the prepared models folder from this Drive link: [Google Drive models folder](https://drive.google.com/drive/folders/1XXANV6cygKAgZ1gX8A1ANh7BorMOeSzy?usp=sharing)
  - After download, place the `models` directory in the project root so that the structure looks like:
    ```
    ./models/faster-whisper/<size>/{config.json, model.bin, tokenizer.json, vocabulary.txt}
    ```

- CLI using huggingface_hub (no code changes):
  - Use the `huggingface_hub` CLI to pre-download models into `./models/faster-whisper`:
    ```bash
    pip install huggingface_hub
    # tiny, base, small, medium, large-v2 (download any you need)
    huggingface-cli download Systran/faster-whisper-tiny --local-dir models/faster-whisper/tiny
    huggingface-cli download Systran/faster-whisper-base --local-dir models/faster-whisper/base
    huggingface-cli download Systran/faster-whisper-small --local-dir models/faster-whisper/small
    huggingface-cli download Systran/faster-whisper-medium --local-dir models/faster-whisper/medium
    huggingface-cli download Systran/faster-whisper-large-v2 --local-dir models/faster-whisper/large-v2
    ```

- Python one-liner (programmatic prefetch):
  ```python
  from huggingface_hub import snapshot_download
  for size in ["tiny","base","small","medium","large-v2"]:
      repo=f"Systran/faster-whisper-{size}"
      snapshot_download(repo_id=repo, local_dir=f"models/faster-whisper/{size}")
  ```

- Configure a custom models directory (optional):
  - Set environment variable before running:
    - Windows PowerShell:
      ```powershell
      $env:WHISPER_LOCAL_DIR = (Resolve-Path "models/faster-whisper").Path
      python app.py
      ```
    - Linux/Mac:
      ```bash
      export WHISPER_LOCAL_DIR="$(pwd)/models/faster-whisper"
      python app.py
      ```

- Preload from the Web UI (Live tab):
  - Use “Preload Selected” or “Preload All” to fetch models into `./models/faster-whisper` without starting recognition.

Notes:
- Model sizes: tiny < base < small < medium < large-v2 (speed vs. accuracy).
- Disk space: each size can range from ~70MB (tiny) up to several GB (large-v2).
- First download can be slow depending on your network and selected size.

5. Set up environment variables:

First, get your Google Gemini API key:
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click on "Get API key"
4. Create a new API key or select an existing one
5. Copy your API key

Then create a `.env` file in the project root with the following variables:
```env
GOOGLE_API_KEY=your_gemini_api_key  # Paste your API key here
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
The application uses Faster-Whisper models which will be downloaded automatically when first used, or you can download them manually using the provided models.zip file. Once downloaded, the models will be stored in the `models/faster-whisper` directory.

In the web UI Live tab you can:
- **Preload Selected**: Use a specific Faster-Whisper model size (tiny/base/small/medium/large-v2)
- **Preload All**: Load all supported model sizes at once
- **Load ASR**: Switch between models instantly

Models will be stored in `./models/faster-whisper/<size>`. After the initial download, the system will use these local files for subsequent runs.

### Model Selection Guide

- **tiny**: Fastest, lowest accuracy
- **base**: Good balance of speed and accuracy
- **small**: Better accuracy
- **medium**: High accuracy
- **large-v2**: Highest accuracy

## Project Structure

```
├── .env                  # Environment variables configuration
├── .env.example          # Example environment variables template
├── .gitignore           # Git ignore rules
├── app.py               # Web interface using Gradio
├── main.py              # CLI application entry point
├── README.md            # Project documentation
└── models/              # Pre-downloaded model directory
    └── faster-whisper/  # ASR models
        ├── base/        # Base model files
        ├── large-v2/    # Large v2 model files
        ├── medium/      # Medium model files
        ├── small/       # Small model files
        └── tiny/        # Tiny model files
```

## System Architecture

- Speech recognition processes audio in real-time at 16kHz
- Voice Activity Detection uses 20ms frame size
- Playback operates at 24kHz for optimal quality
- System requirements depend on the chosen Whisper model size

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## License & Copyright

This project was developed as an assignment for Word Works AI India Pvt Ltd. All rights are reserved by Word Works AI India Pvt Ltd. This code is proprietary and confidential. Unauthorized copying, modification, distribution, or use of this code, via any medium, is strictly prohibited without express written permission from Word Works AI India Pvt Ltd.

© 2025 Word Works AI India Pvt Ltd. All Rights Reserved.

## Acknowledgments

- Faster Whisper for efficient speech recognition
- Google Gemini for natural language processing
- Microsoft Edge TTS for high-quality speech synthesis
- WebRTC VAD for voice activity detection
