"""
session.py
----------
Guided end-user session flow layered on top of the four model stages:
consent and disclaimer, the stimulus (handled by emotion_detector, called
from pipeline.py), three spoken questions, processing, profile, chat, and
a closing line with professional-help resources.

This is the piece the Design chapter describes - informed consent, local
processing, signposting to real help - that previously only existed as
prose. Before this module, pipeline.py ran as a bare developer script: it
jumped straight from webcam capture to a single microphone recording, with
no consent step, no structured questions, and no closing message. That gap
mattered for more than completeness - the ethics argument made in Design
only counts as implemented if it exists in the code a reviewer or examiner
can read, and running a session on a real tester (the peer review and user
study both require this) isn't defensible without a consent step.

Kept deliberately terminal-based, in line with the rest of the prototype:
a clear guided flow printed to the terminal (plus the existing OpenCV
window from Stage 1) is enough to demonstrate consent, structured
questioning, and closing signposting. Building a GUI is explicitly out of
scope for this project (see work plan, scope control).
"""

CONSENT_TEXT = """
============================================================
 Emotional AI Mirror - before we start
============================================================
This session will:
  - turn on your webcam and analyse your facial expressions
    while you watch a short sequence of images
  - record your microphone while you answer three spoken
    questions, and transcribe what you say
  - use both to generate a short, non-clinical reflection on
    your emotional patterns, which you can then discuss

Please also note:
  - this tool does NOT diagnose any medical or psychiatric
    condition, and is not a replacement for professional care
  - all processing happens locally on this machine - nothing
    is uploaded or sent to a server
  - your recording and the session transcript are saved to a
    local log file on this machine so the session can be
    reviewed later; nothing leaves this device
  - you can stop at any point by closing the webcam window or
    pressing Ctrl+C

You must be 18 or older to use this tool.
============================================================
"""

# The three spoken questions asked after the stimulus, in order. Kept as a
# module-level constant (rather than buried in a function) so pipeline.py,
# the report, and any future variant of this flow all read from one place.
SPOKEN_QUESTIONS = [
    "How are you feeling right now?",
    "What did those images bring to mind?",
    "Is anything weighing on you?",
]

# Deliberately not exhaustive or country-specific beyond a couple of
# well-known examples - the point of this message is "here is where to
# look", not to be a verified, jurisdiction-complete directory. Anyone
# deploying this beyond a coursework prototype should review and localise
# this list rather than trust it as-is.
RESOURCES_TEXT = """
------------------------------------------------------------
This reflection is not a diagnosis, and this tool is not a
substitute for professional support. If anything that came up
today feels like more than you want to sit with alone, it's
worth talking to a doctor or a mental health professional.

If you're in the UK: Samaritans, 116 123, free, 24/7.
If you're in the US: 988 Suicide & Crisis Lifeline.
Elsewhere: Befrienders Worldwide (https://www.befrienders.org)
lists local helplines by country.
------------------------------------------------------------
"""


def get_consent(auto_yes=False):
    """Print the consent/disclaimer text and ask the user to confirm.

    auto_yes=True skips the interactive prompt - used by --demo runs and
    other non-interactive walkthroughs (e.g. a scripted peer-review demo)
    where a real person has already agreed to run the demo out of band,
    rather than this specific session.

    Returns True only on an explicit 'yes' (or auto_yes=True). Anything
    else - a blank line, 'no', an interrupt, EOF - is treated as consent
    withheld, since an unclear answer here must never be read as agreement.
    """
    print(CONSENT_TEXT)
    if auto_yes:
        print("(--demo mode: consent auto-confirmed for this walkthrough)\n")
        return True

    try:
        answer = input("Type 'yes' to continue, or press Enter to exit: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer == "yes"


def run_spoken_questions(transcriber, use_sample=False, sample_audio_path=None, mic_dir=None):
    """Ask each of the three spoken questions in turn, recording (or, in
    demo mode, transcribing the bundled sample) a separate answer for each.

    Returns a list of {"question": ..., "answer": ...} dicts, in the order
    asked, so both the combined transcript and the session log can show
    which answer responded to which question.

    In demo mode all three questions are transcribed from the same bundled
    sample.wav, since only one sample file ships with the project - this is
    a known limitation of --demo mode, not a bug, and is printed to the
    user so it isn't mistaken for three distinct real answers.
    """
    from recorder import record_audio  # local import: matches the lazy-import pattern used elsewhere

    qa_pairs = []
    for i, question in enumerate(SPOKEN_QUESTIONS, start=1):
        print(f"\nQuestion {i}/3: {question}")

        if use_sample:
            print("(--demo mode: reusing the bundled sample.wav for every question)")
            recording_path = sample_audio_path
        else:
            recording_path = record_audio(mic_dir / f"session_q{i}.wav")

        transcription = transcriber.transcribe(str(recording_path))
        print(f"You said: {transcription.text}")
        qa_pairs.append({"question": question, "answer": transcription.text})

    return qa_pairs


def format_qa_transcript(qa_pairs):
    """Combine the three question/answer pairs into the single transcript
    string the sentiment and language stages both expect. Keeping the
    question visible alongside each answer (rather than just concatenating
    the three answers) lets the language model see what each answer was
    responding to, rather than three unlabelled sentences run together."""
    return "\n".join(f"Q: {pair['question']}\nA: {pair['answer']}" for pair in qa_pairs)


def print_closing(profile):
    """Print the closing line and resources.

    Always shown, not only when profile.seek_help is set - a closing
    signpost to professional support belongs at the end of every session,
    per the Design chapter's severity-assessment and ethics arguments, not
    only the sessions the model happens to flag as concerning.
    """
    print("\n" + "=" * 60)
    print("Thank you for taking part in this reflection.")
    if profile.seek_help:
        print("\nBased on what came up today, this is a good moment to")
        print("consider talking to a professional - see below.")
    print(RESOURCES_TEXT)
