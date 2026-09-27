# CS_Final_Project

# Emotional AI Mirror

A privacy-focused proof of concept for emotional self-reflection, built for the University of London BSc Computer Science final project (CM3070, Template 4.1: Orchestrating AI models to achieve a goal).

The system runs a guided session: the user watches a short emotional image sequence while their facial expressions are sampled, answers three spoken questions, and receives a non-diagnostic reflection they can discuss in a follow-up chat. Four pre-trained models are orchestrated, all running locally:

| Stage | Model |
|---|---|
| Facial emotion | DeepFace |
| Speech-to-text | faster-whisper (tiny) |
| Text emotion | j-hartmann/emotion-english-distilroberta-base |
| Reflection and chat | llama3.2 via Ollama |

**This tool does not diagnose any condition and is not a substitute for professional care.**

## Requirements

- Python 3.12
- [Ollama](https://ollama.com) installed and running
- Webcam and microphone (not needed in `--demo` mode)
- Tested on Windows 11, Intel Core i7-1185G7, 16 GB RAM, no dedicated GPU

## Installation

```
git clone https://github.com/leonardo-bracci/CS_Final_Project.git
cd CS_Final_Project
python -m venv .venv
.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2
```

The DeepFace, faster-whisper and j-hartmann model weights download automatically on first run.

## Usage

Full session (webcam, microphone, consent prompt):
```
python src/pipeline.py
```

Demo mode (bundled sample audio, no webcam stimulus, consent auto-confirmed):
```
python src/pipeline.py --demo
```

Type `quit` to end the chat. Each session writes a JSON log and a timing log to `logs/`.

## Tests

```
python -m pytest
```

56 unit tests; hardware and models are mocked, so no webcam, microphone or Ollama is needed.

## Evaluation scripts

| Script | Purpose |
|---|---|
| `scripts/emotion_benchmark.py` | DeepFace vs FER |
| `scripts/audio_benchmark.py` | faster-whisper vs openai-whisper |
| `scripts/sentiment_benchmark.py` | VADER vs j-hartmann vs tabularisai |
| `scripts/ollama_benchmark.py` | llama3.2 vs qwen2.5:3b |
| `scripts/chat_robustness_test.py` | 10-message chat robustness test |
| `scripts/eval_seek_help.py` | Re-scores saved logs with the seek-help rule |
| `scripts/plot_emotion_timeline.py` | Per-frame emotion timeline plot |

## Project structure

```
src/        pipeline, session flow, and one module per model stage
tests/      pytest suite
scripts/    benchmarks and evaluation
samples/    sample audio and OASIS stimulus images
reports/    project reports
```

## Privacy and safety

- All processing is local; nothing is sent to a server.
- Session logs are stored locally in `logs/` and are excluded from this repository.
- The seek-help flag is computed in code from the user's own words, not by the language model.
- Support resources are shown at the end of every session.

## Credits

Stimulus images from the Open Affective Standardized Image Set (OASIS): Kurdi, B., Lozano, S., & Banaji, M. R. (2017). *Behavior Research Methods*, 49(2), 457–470.