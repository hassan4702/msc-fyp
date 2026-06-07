# Evaluation

Two evaluation tracks, matching the research questions.

## Quantitative (RQ1, RQ2) — on MELD

`evaluate_meld.py` (planned): for each MELD utterance, run the text branch on the
transcript and the face branch on the video frames, then compare four systems —
text-only, face-only, fused, and majority-class — on **macro-F1 + per-class F1 +
confusion matrices**. The fusion weights / learned arbiter are tuned on MELD-train
and reported on MELD-test. Conflict-case analysis (RQ2) lives here too.

## Qualitative (RQ3) — user study

`analyse_study.py` (planned): paired within-subject design (each participant uses
both the multimodal and text-only bot, order counterbalanced), Likert ratings for
empathy / appropriateness / understanding, analysed with the **Wilcoxon
signed-rank test** (`scipy.stats.wilcoxon`).

Raw study data is sensitive and **must not** be committed (see `.gitignore`).
