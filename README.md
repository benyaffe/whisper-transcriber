# Whisper Transcriber

A macOS GUI app for transcribing audio/video files to VTT/TXT using faster-whisper.

## Features

- **Drag & Drop**: Drop files onto the app icon or dashboard
- **URL Downloads**: Paste YouTube, Instagram, TikTok links
- **Smart Quality**: Auto-upgrades model if quality is low
- **Live Preview**: See transcription in real-time
- **Batch Processing**: Queue multiple files

## Requirements

- macOS 10.15+
- Python 3.10+
- FFmpeg (`brew install ffmpeg`)

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python main.py
```

## Building the App Bundle

```bash
# Development build (uses system Python)
python setup.py py2app -A

# Production build (standalone)
python setup.py py2app

# App will be in dist/Whisper Transcriber.app
```

## Usage

1. **Local Files**: Drag audio/video files onto the window, or click to browse
2. **URLs**: Paste a video URL and click "Add URL"
3. **Monitor**: Watch progress and live transcript preview
4. **Output**: VTT and TXT files are saved next to the source file

## Output Formats

- **VTT**: WebVTT with timestamps (great for search/reference)
- **TXT**: Clean prose without timestamps

## Speaker Identification (Optional)

Identifies different speakers in your audio using pyannote.audio. Requires a free HuggingFace account:

1. Create an account at [huggingface.co](https://huggingface.co/join)
2. Accept the license for all three required models:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
3. Create an access token at [Settings > Access Tokens](https://huggingface.co/settings/tokens) with **Read** permission
4. Enter the token in the app's Settings dialog

## Model Selection

Starts with `medium` model. If confidence is low after 120 seconds, automatically restarts with `large` model.
