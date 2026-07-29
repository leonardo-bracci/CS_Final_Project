Emotional AI Mirror — Work Plan

Rebuilt 29 July 2026. One person, code + report. Currently Week 15.

Topics 1-7 are complete (Preliminary Report, Topic 6 write-up, Topic 7 Testing peer review all submitted). What remains: finish the build, second peer review, draft report, final report, public repo, demo video.

Three confirmed dates:

Draft report: 17 Aug, 3:00 PM CEST (19 days out, falls on a Sunday, so the last full working day is Friday 15 Aug)
Online exam (Inspera IEP): 15 Sep, reflection exam, not coding
Final report + video: 28 Sep, 3:00 PM CEST, hard deadline

That leaves about 2.5 weeks, full-time, to get the four-model pipeline running end to end and draft six chapters. The build needs to stop once the pipeline works, not once it's polished. The draft report does not need a finished project - it's expected to be in progress, and exists to get tutor feedback.

0. Current state

Built:

Stage 1, vision: emotion_detector.py (webcam capture, DeepFace every 10th frame, scene tagging, timeline summary, CSV export)
Stage 2, audio: transcriber.py (faster-whisper wrapper; the Whisper to faster-whisper swap is already documented with justification)
pipeline.py, orchestrating Stages 1-2 only

Submitted: Topic 6 prototype write-up, Topic 7 Testing peer review.

Not yet built:

Stage 3, sentiment/emotion-from-text - the next blocker, needs to be done first
Stage 4, Ollama profile generation + interactive chat
Live microphone recording (pipeline currently transcribes a bundled sample.wav)
Real stimulus content (scenes are currently placeholder labels)
Guided session flow (consent, stimulus, questions, processing, profile, chat)
Unit-test harness
Second peer review, draft report, final report, video, public repo
1. Deadline map
Deliverable	Date	Status
Draft report	17 Aug, 3:00 PM CEST	Confirmed, 19 days out
Topic 9 peer review (Testing 2)	after draft, roughly late Aug	Need to confirm portal date
Online exam (Inspera IEP)	15 Sep	Confirmed
Code repository public	before 28 Sep	Tie to final report
Final report + video	28 Sep, 3:00 PM CEST	Confirmed, hard deadline

Only the second peer-review date is still unconfirmed - check the portal.

2. The next 19 days: build then draft

Roughly 9 days of build (30 Jul-8 Aug), then 7 days of writing (9-15 Aug), submit 17 Aug. Build stops once the four models run end to end; it doesn't need to be polished at that point.

Phase A, 30 Jul-1 Aug: Stage 3, emotion from text

Test three candidates on the same ~10 sample transcripts:

j-hartmann/emotion-english-distilroberta-base - uses the same 7 labels as DeepFace (anger, disgust, fear, joy, neutral, sadness, surprise)
tabularisai/multilingual-sentiment-analysis - 5-class sentiment, the model the template links
VADER - rule-based baseline, near-zero cost

Log per model: latency per input, RAM use, label quality, and put it in one comparison table - this is the "testing and rejecting models" evidence the report needs.

The design argument worth capturing the day it's decided: a text model that shares DeepFace's label space lets the LLM compare face vs. speech directly (e.g. "your face showed sadness while your words expressed joy"). Plain sentiment can't support that comparison.

Build sentiment.py following the Transcriber pattern (lazy import, load once, return a small structured result), then wire it into pipeline.py as Stage 3.

Start pytest alongside: summarise_timeline, get_current_scene, and sentiment mapping, all mocked so it runs offline.

Exit condition: one command runs webcam through transcript through text-emotion scores, and the comparison table is saved.

Phase B, 2-4 Aug: Stage 4, Ollama profile + chat

Build profile_generator.py, wrapping the ollama client. Structured prompt: per-scene facial emotions + transcript + text emotions, producing a profile with observed patterns, focus areas, wellbeing recommendations, and a professional-help flag.

Add a chat loop with history for follow-ups.

System-prompt rules: supportive, never diagnoses, frames output as self-reflection, signposts professional help when the severity flag is set. Ezell (2026) is the published justification for this.

Compare at least two local models (llama3.2:3b vs. qwen2.5:3b or phi3:mini) on time-to-first-token, total generation time, and output quality. This becomes a second evidence table.

Add mocked-Ollama unit tests: prompt assembly, severity-flag logic, empty-input handling.

Exit condition: the full four-model chain runs end to end on sample inputs, producing a profile and a working follow-up chat.

Phase C, 5-6 Aug: live mic + real stimulus

Build recorder.py using sounddevice: press-to-stop capture, save as WAV, feed into Transcriber. Write the three spoken questions: "How are you feeling right now?", "What did those images bring to mind?", "Is anything weighing on you?"

Replace the placeholder scenes with real stimulus content. The cheapest defensible route is the OASIS open affective image set (free; IAPS needs research access), sequenced into timed scenes using the same structure as the current video_sequence.

Exit condition: speaking into the mic produces a transcription and classification, and the stimulus is real images rather than labels.

Phase D, 7-8 Aug: session flow, demo mode, logging

Build the guided session: consent and disclaimer, stimulus with webcam analysis, three spoken questions, processing, profile, chat, and a closing line with professional-help resources. A terminal plus OpenCV window is sufficient for this - the goal is a clear flow, not a polished interface.

Add a --demo mode: runs the whole pipeline on a bundled sample video and audio, without needing a webcam, mic, or waiting on Ollama. This makes the second peer review realistic and is worth documenting as a deliberate engineering decision.

Log every session to a timestamped JSON file (all four signals plus outputs) - this becomes Evaluation data later.

Exit condition: someone with only the README could run a full session, or the demo, end to end. Build freezes after this phase; anything left is post-draft polish.

Phase E, 9-15 Aug: write the draft (7 days)

Write Implementation first while the code is still fresh. Chapter caps for the draft: Introduction 1000, Literature review 2500, Design 2000, Implementation 2000, Evaluation 2500, Conclusion 1000, total 9500.

Implementation (~2000 words): architecture diagram, then per stage - what it does, one key code excerpt explained, why that model was chosen (referencing the two comparison tables), and screenshots.
Evaluation (~2000-2500 words): unit-test results, per-model benchmarks, pipeline timing, known failure modes (e.g. DeepFace's neutral-to-fear/disgust bias), Topic 7 peer-review results, and any pilot user results gathered by this point.
Introduction (~1000 words): revised from the Preliminary Report, stating the template number.
Literature review (up to 2500 words): the instructions say this revises the second peer-review chapter (the literature review specifically), not just the Preliminary Report, so pull that exact version. Add only new sources actually used (stimulus set, emotion-model paper).
Design (up to 2000 words): revises the third peer-review chapter (the design). Add the final pipeline/flow diagrams; the four existing justifications still hold. Add an inclusive-design paragraph - see note below, this is currently missing and worth marks both in the report and the exam.
Conclusion (~800 words): summary plus further work.

Write in completed tense throughout, except for genuine future work. Every claim should be backed by a citation, a number, or a figure.

Inclusive design is a real gap right now. The syllabus has a dedicated inclusive-design topic and the exam weights it 12 marks (Q2b/c), but nothing in the current design covers it. This isn't a box-ticking exercise: facial-emotion models have documented demographic accuracy bias by skin tone and gender; Whisper's transcription accuracy varies by accent and language; the interface (terminal vs. GUI, colour, text size) has accessibility implications; and the project's low-cost, stigma-reducing framing is itself an inclusion argument already being made elsewhere in the report. A short, honestly justified paragraph in Design plus a stated limitation in Evaluation covers this, and doubles as preparation for exam question Q2c.

17 Aug, 3:00 PM CEST: draft due

Submit up to 9500 words, respecting every chapter cap. Submit even if the system isn't finished - the draft's purpose is to get free tutor feedback before the document is resubmitted as the final report. It's a Sunday deadline, so don't leave submission until the last few minutes.

3. After the draft (18 Aug to 28 Sep)
~18-31 Aug: second peer review, user testing, iteration

Prepare the Topic 9 peer-review package: three refreshed questions covering the new stages, an updated build and README, and a tagged release. (Confirm the portal date for this - it may fall inside this window.)

Run the 6-10 tester study: consent sheet, pre-session self-report, full session, then a 1-5 Likert survey on accuracy, clarity, and usefulness.

Analyse the results (mean score per question, self-report vs. profile agreement, representative quotes), fix the top two or three issues found, and document what changed before and after. Testing followed by iteration, with evidence, is one of the clearest ways to separate a good submission from a top one.

Draft Topic 9 questions:

"The conversation responses related sensibly to my profile."
"The system made clear it was not giving a medical diagnosis."
"The full session ran from start to finish without technical problems."
~1-13 Sep: final report + record the video

Aim to have the substantive work done by around 13 Sep.

Work the tutor's draft feedback into both the report and the code, keeping a one-line-per-item change log.

Expand Implementation to the final 2500-word cap, finish Evaluation with the full user study and iteration story, and bring the whole report within the 10,500-word final limit.

Clean up the repository: clear README, requirements.txt, bundled sample data, remove dead files, switch the repo to public, and verify the link works while logged out.

Record the video before the exam, while the build is still stable. Aim for a script of roughly 450-600 words (3-4 minutes): problem and who it's for (~20s), an uninterrupted full-session demo (~2-2.5 min), the four models and why each was chosen (~60s), and evaluation highlights plus one honest limitation (~30s). Use real screen footage, your own voice, no sped-up sections, and stay within 3-5 minutes.

15 Sep: online exam (Inspera IEP)

This is a reflection exam, not a coding task, prepared from the evidence log. Q1 is compulsory (title, template, what the project was about) and scores zero for the whole paper if left blank. Marked questions:

Q2a (8 marks): how the project related to Template 4.1 - flexibility, how it was used, tradeoffs between flexibility and constraint.
Q2b/c (12 marks): inclusive design and radical inclusion, applied specifically to this project. Prepare this while writing the Design chapter.
Q3a (8 marks): final outcomes vs. initial aims - what was met, and honest reasons for any gaps.
Q3b (12 marks): three things that would be done differently, with weaknesses and fixes.
Q4 (20 marks): whether the background reading was appropriately scoped, and early time-allocation decisions.

Preparation is one pass over the evidence log (section 5) the week before - there should be nothing left to cram if the log has been kept current.

16-27 Sep: final polish

Do a rubric pass: template number and project number present in the Introduction, all word counts under their caps, the repo link working while logged out, video within 3-5 minutes.

28 Sep, 3:00 PM CEST: final report + video due

Submit the report (up to 10,500 words), a public repo link, and the video. Late submissions and submissions over the word limit are penalised - leave margin.

4. Grade logic
Pass (40+): a working system with at least three pre-trained models across different data types, a clear goal, and evidence of model evaluation. Phases A-D deliver a four-model system spanning vision, audio, text, and language.
50-69: a challenging goal that genuinely needs pre-trained models, integrated software, and both unit and user testing. The "why pre-trained models" argument should be explicit - real-time facial emotion recognition, open-vocabulary transcription, and adaptive supportive dialogue are each impractical to hand-code with rules.
First class: thorough exploration of the model space with evidence (compare-and-choose at all four stages, starting from the Whisper-to-faster-whisper swap), deliberate interaction design (guided flow, consent, demo mode), and user testing that visibly changed the design.

Things that lose marks, worth checking before every submission:

Word limits are strict and penalised, both per section and overall. Implementation is 2000 words in the draft but 2500 in the final.
Future tense reads as work not yet done - write in completed tense.
For the video: no working demo, an AI-generated voice, sped-up footage, or running outside 3-5 minutes are all penalised.
The repo not being public, or the link being broken, at the time of submission.
Missing the template number in the Introduction.
5. Scope control - what not to do
No web framework or GUI. A clean guided flow in the terminal scores fine; a polished interface isn't required.
No model families beyond the planned comparisons. Two or three candidates per stage is thorough; five starts to look like avoidance of the writing.
No accounts, history dashboards, or mood tracking - these belong in the further-work section, not the build.
Build freezes 8 Aug. The writing days (9-15 Aug) should not be touched by further coding. A rougher system with a strong draft is a better outcome than a more polished system with a rushed one.
6. Ethics guardrails

The system never diagnoses; output is framed as self-reflection and wellbeing suggestions, and the severity flag routes to a recommendation to speak to a professional, with resources. Ezell (2026) and Riedl et al. (2024) support this framing.

Testers sign a short consent sheet; recordings stay on-device and are deleted after the study.

These points belong in both Design and Evaluation - "justification with evidence" is a named marking criterion.

7. Evidence log

Keep one folder with one file per item: per-stage candidates, benchmarks, and decisions; screenshots of each stage working for the first time; a bug list with fixes; peer-review responses and what changed as a result; tutor draft feedback and how it was addressed; user-study consent sheets, raw data, and analysis.

