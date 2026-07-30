# Emotional AI Mirror — Work Plan

**Rewritten 30 July 2026 · one person, code + report · 18 days to the draft**

Everything below is work still to do, in the order it should be done. The build is close enough to complete that writing days are now the scarce resource, not coding days — so the plan protects them.

## Fixed dates

| Deliverable | Date | Notes |
|---|---|---|
| **Draft report** | **Sun 17 Aug, 3:00 PM CEST** | Sunday deadline → treat **Fri 15 Aug** as the real one |
| Topic 9 peer review (Testing 2) | late Aug — **date unconfirmed** | Check the portal this week; it may land inside the user-study window |
| **Online exam (Inspera IEP)** | **Tue 15 Sep** | Reflection exam, no coding |
| **Final report + video** | **Mon 28 Sep, 3:00 PM CEST** | Hard deadline, repo must be public and live |

**Build freeze: Fri 7 Aug, end of day.** After that, nothing is written except the report — a rougher system with a strong draft scores better than the reverse, and the draft is explicitly expected to describe work in progress.

---

## 1. Priority order

Ranked by marks-per-hour, not by what's most interesting. Work down the list; if time runs short, the things at the bottom are the things to drop.

### P1 — Required, unwritten, and needs no one else's cooperation

1. **Inclusive design paragraph (Design chapter).** The largest single gap in the report. It is also exam Q2b/c at 12 marks, so it is written once and used twice. Cover honestly: demographic accuracy bias in facial-emotion models by skin tone and gender; Whisper accuracy variation by accent and language; accessibility of a terminal + OpenCV interface; and the fact that the low-cost, stigma-reducing framing is itself an inclusion argument. One or two tight paragraphs beat a padded page.
2. **References section.** There is currently no reference list at all, while a dozen in-text citations run through the report and "proper citation and referencing" is its own marking criterion. This is free marks and about two hours of work. Add the missing sources at the same time: DeepFace, faster-whisper, j-hartmann, the Brunel paper (properly cited, not "a 2025 study from Brunel"), and the stimulus set with its licence.
3. **Implementation evidence: four code snippets + architecture diagram + screenshots.** Implementation is capped at 2000 words in the draft and is the chapter most at risk of being under-evidenced. Push everything possible into figures, which cost no words: architecture diagram, the OpenCV window mid-session, pytest passing, both benchmark outputs, a real generated profile, a chat exchange referring back to it.
4. **Conclusion chapter.** Not started. Short, and the one chapter where forward-looking language is expected — worth a first pass purely to get tutor feedback on framing.

### P2 — Small build items, each unlocking report content it can't be written without

5. **Guided session flow.** Consent and disclaimer → stimulus → three spoken questions → processing → profile → chat → closing line with professional-help resources. This is the widest gap between what Design promises and what the code does, it is a top-band criterion ("consider the best way to design the user interaction"), and the ethics argument only counts if it exists in the build. It is also a precondition for running user testing on a stranger. Terminal + OpenCV window is sufficient; do not build a GUI.
6. **Real stimulus content.** Replace the four placeholder scene labels. OASIS is the cheapest defensible route — free, openly licensed, validated valence/arousal ratings, no research-access application (IAPS needs one). Sequence into timed scenes using the existing `video_sequence` structure. Record the selection criteria as you go; that reasoning is report content.
7. **DeepFace vs at least one alternative.** Vision is the only stage with no compare-and-reject evidence, in a report where the other three all have it, and the brief names that process explicitly. Benchmark against FER, py-feat, or an ml5/MediaPipe face model on one recorded clip: per-frame latency, FPS impact, and behaviour on a resting face. That last measure doubles as a test of whether the neutral-to-fear/disgust bias is DeepFace-specific or general.
8. **Structured chat robustness test.** Five to ten varied and ambiguous follow-up messages through the chat loop, logging how often replies are off-topic, inconsistent, or produce an inappropriate refusal. Repeat on qwen2.5:3b. Two erratic-response incidents are already documented; this turns an anecdote into a finding, and a finding into a defensible model choice.
9. **End-to-end pipeline timing.** The Evaluation methodology commits to it and no number is reported. Time one full session broken down by stage. The honest result is likely that local-only processing costs a real wait — naming that tradeoff scores better than omitting it.

### P3 — Needs other people, so start it early

10. **Pilot user study, 2–3 people, before the draft.** The full 6–10 person study belongs after the draft, but an empty User Testing heading is the weakest thing a tutor can see. A small pilot with a stated protocol (consent sheet, pre-session self-report, 1–5 Likert on accuracy/clarity/usefulness) fills the section, tests the instrument before it's used properly, and earns feedback on the study design while there's still time to act on it.
11. **Write up the Topic 7 peer review.** Already completed, currently absent from Evaluation. Report the three questions, the responses, and what changed because of them. Feedback followed by a documented change is the clearest evidence of iteration.

### P4 — After the draft (see section 3)

Full user study · Topic 9 peer review · final report expansion · video · public repo · exam.

---

## 2. Day map to 17 Aug

**Thu 30 Jul – Sun 2 Aug · build sprint**
Guided session flow with consent and the three spoken questions (P2.5). Real stimulus in (P2.6). Nothing else — these two are the only items that change what the system *is*.
*Exit: a stranger can be sat down in front of it and taken through a complete session.*

**Mon 3 – Wed 5 Aug · measurement sprint**
DeepFace comparison (P2.7). Chat robustness test on both models (P2.8). End-to-end timing (P2.9). Capture every screenshot and graph listed in P1.3 while the system is in front of you — going back for a missing screenshot during the writing week is pure waste.
*Exit: three new result sets saved, and a folder of figures ready to drop into the report.*

**Thu 6 – Fri 7 Aug · pilot + freeze**
Run the 2–3 person pilot (P3.10). Fix only what the pilot breaks. **Build freezes Friday end of day.**

**Sat 8 – Fri 15 Aug · writing, in this order**

| Days | Chapter | Notes |
|---|---|---|
| 8–9 Aug | **Implementation** (2000) | Write first, while the code is freshest. Snippets, architecture diagram, session-flow section, screenshots |
| 10–11 Aug | **Evaluation** (2500) | New results from the measurement sprint, Topic 7 peer review, inclusive-design limitations, pilot results |
| 12 Aug | **Design** (2000) | Inclusive design paragraph, final pipeline and flow diagrams |
| 13 Aug | **Introduction** (1000) + **Literature review** (2500) | Add the project number to the Introduction. Lit review pulls the second peer-review version and adds only sources now actually used |
| 14 Aug | **Conclusion** (1000) + **References** | |
| 15 Aug | **Full pass** | Word count per chapter and overall (≤9500), every TODO and grey note deleted, tables and figures numbered in order and cited by number in the prose, completed tense throughout |

**Sat 16 Aug · submit.** A full day early. The deadline is 3:00 PM on a Sunday; nothing good happens by leaving it to Sunday afternoon.

Write in completed tense except for genuine future work. Every claim carries a citation, a number, or a figure.

---

## 3. After the draft: 18 Aug → 28 Sep

**Week of 18 Aug — Topic 9 peer review package.** Confirm the portal date first. Three refreshed questions covering the stages that didn't exist at Topic 7:
- "The conversation responses related sensibly to my profile."
- "The system made clear it was not giving a medical diagnosis."
- "The full session ran from start to finish without technical problems."

Submit a tagged release with a `--demo` build that runs without webcam, microphone or Ollama, plus written instructions stating exactly what software and hardware are required.

**~20 Aug – 7 Sep — full user study and iteration.** 6–10 testers: consent sheet, pre-session self-report, full session, 1–5 Likert survey. Analyse (mean per question, self-report vs. profile agreement, representative quotes), fix the top two or three issues found, and document the before and after. **Testing that visibly changed the design is the clearest separator between a good submission and a top one** — budget time for the change, not just the measurement.

**8–13 Sep — final report and video.**
- Work the tutor's draft feedback into report and code, keeping a one-line-per-item change log.
- Expand Implementation to its final 2500 cap; complete Evaluation with the full study and the iteration story; bring the whole report under 10,500.
- Repo: clear README, `requirements.txt`, bundled sample data, dead files removed, **switch to public**, verify the link while logged out.
- **Record the video before the exam**, while the build is stable. ~450–600 word script, 3–4 minutes: problem and who it's for (20s) → uninterrupted full-session demo (2–2.5 min) → the four models and why each was chosen (60s) → evaluation highlights plus one honest limitation (30s). Real screen footage, own voice, no speed-ups, inside 3–5 minutes.

**15 Sep — online exam (Inspera IEP).** Preparation is one pass over the evidence log the week before; nothing to cram if the log is current. **Q1 is compulsory and a blank answer scores zero for the entire paper** (title, template, what the project was about). Marked questions:

| Q | Marks | Prepare from |
|---|---|---|
| Q2a | 8 | How the project related to Template 4.1 — flexibility, how it was used, flexibility-vs-constraint tradeoffs |
| Q2b/c | 12 | Inclusive design and radical inclusion, applied to this project — **written during P1.1, not from scratch** |
| Q3a | 8 | Final outcomes vs. initial aims, with honest reasons for gaps |
| Q3b | 12 | Three things you'd do differently, with weaknesses and fixes |
| Q4 | 20 | Whether background reading was appropriately scoped; early time-allocation decisions |

**16–27 Sep — final polish.** Rubric pass: template *and* project number in the Introduction, every word count under its cap, repo link working logged-out, video inside 3–5 minutes.

**28 Sep, 3:00 PM CEST — submit** report (≤10,500), public repo link, and video. Leave margin.

---

## 4. Standing rules

**Scope control — what not to do.** No web framework or GUI. No model families beyond the comparisons already planned — two or three candidates per stage is thorough, five is avoidance of the writing. No accounts, history dashboards or mood tracking; those are further-work section material. Build freezes 7 Aug and the writing days stay untouched by code.

**What separates the bands from here.** A pass needs the working four-model system with evaluation evidence, which effectively exists. 2:1 territory needs integrated software plus unit *and* user testing, and an explicit argument for why pre-trained models were necessary at all — real-time facial emotion recognition, open-vocabulary transcription and adaptive supportive dialogue are each impractical to hand-code. A first needs compare-and-choose evidence at **all four** stages (hence P2.7), deliberate interaction design (hence P2.5), and user testing that visibly changed the design (hence P3.10 and the September study).

**Mark-losers, check before every submission.** Word limits are strict and penalised per section as well as overall — Implementation is 2000 in the draft but 2500 in the final. Future tense reads as work not done. Video: no working demo, an AI-generated voice, sped-up footage, or running outside 3–5 minutes are each penalised. Repo not public or link broken at submission. Missing template or project number in the Introduction.

**Ethics guardrails** — these belong in both Design and Evaluation, since "justification with evidence" is a named criterion. The system never diagnoses; output is framed as self-reflection and wellbeing suggestions; the severity flag routes to a recommendation to speak to a professional, with resources. Ezell (2026) and Riedl et al. (2024) support the framing. Testers sign a short consent sheet; recordings stay on-device and are deleted after the study.

**Evidence log — update it the day things happen.** One folder, one file per item: per-stage candidates, benchmarks and decisions; screenshots of each stage first working; bug list with fixes; peer-review responses and what changed; tutor draft feedback and how it was addressed; user-study consent sheets, raw data and analysis.

Double-duty for 15 Sep: add a line whenever you hit a template flexibility or constraint decision (Q2a), an inclusive-design choice or bias limitation (Q2c), an outcome that diverged from the aim (Q3a), something you'd do differently (Q3b), or background reading that did or didn't pay off (Q4).
