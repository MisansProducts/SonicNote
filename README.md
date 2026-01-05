# SonicNote

SonicNote is a Python project that takes a recorded meeting \( \(.wav\) \) and runs:

- Speaker diarization (who spoke when)
- Transcription (what was said)
- LLM summarization (what matters)

## How it works

1. Load a meeting audio file \( \(.wav\) \)
2. Diarize speakers with `pyannote.audio`
3. Transcribe audio (Whisper / Faster-Whisper)
4. Summarize the transcript using an LLM (via `ollama`)

## Current tasks

1. Improve the function that recognizes different speakers (better diarization / speaker labeling)

## Dependencies

### pyannote.audio (diarization)

Install and configure `pyannote.audio` by following the official instructions:  
https://github.com/pyannote/pyannote-audio?tab=readme-ov-file

### Ollama (LLM summarization)

Install Ollama from:  
https://www.ollama.com/

Make sure you have at least one model pulled (for example, you might run `ollama pull qwen2.5`).

## Installation

1. Create and activate a virtual environment (recommended).
2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

### requirements.txt

- faster-whisper  
- ollama  
- openai-whisper  
- pyannote.audio  
- pydub  
- python-dotenv  
- speechrecognition  
- torch  
- torchaudio  
