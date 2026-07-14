# Emotion AI Mirror

Local, privacy-focused emotional self-awareness tool. Combines facial emotion
recognition, speech-to-text, sentiment analysis, and a local LLM (Ollama) into
one pipeline that runs entirely on-device.

Final year project — CM3020 Artificial Intelligence.

## Status
Early prototype: facial emotion recognition + scene tagging + basic Ollama
summary. See `Preliminary_Report_3.pdf` for full design and plan.

## Setup
```bash
pip install -r requirements.txt
ollama pull llama3.2
python emotion_test.py
```
