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
pip install gradio av edge-tts google-generativeai python-dotenv numpy huggingface_hub
```

4. Models Setup:

The Faster-Whisper models can be obtained in two ways:
- **Automatic Download**: The models will be downloaded automatically when you first run the application.
- **Manual Download**: 
  ```bash
  # Download models.zip from: [Add your preferred hosting link here]
  # Then extract it:
  Expand-Archive -Path models.zip -DestinationPath . -Force  # For Windows PowerShell
  # OR
  unzip models.zip  # For Linux/Mac
  ```

Note: Models are downloaded from the Hugging Face Hub. The first run might take some time depending on your internet connection and the model size chosen.

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
