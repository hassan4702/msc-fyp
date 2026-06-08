# MELD Evaluation Results

Run: 2026-06-09. Test split = 2,610 utterances. Arbiter trained on 9,989 train
utterances. Face-detection coverage on test: **88.7%** (the rest are cuts /
profiles / multi-face frames → handled as "no face" → text-only fallback).

**Critical framing — cross-corpus transfer, not in-domain.** The text model was
fine-tuned on **GoEmotions**, the face model on **FER-2013**, then both applied to
**MELD with no MELD fine-tuning**. These numbers measure *generalization* to a new
domain (TV dialogue), which is why absolute macro-F1 is ~0.27 (well below in-domain
MELD text SOTA). In-domain fine-tuning on MELD's train split is the natural follow-up.

## RQ1 — macro-F1 on MELD test (all utterances)

| System | macro-F1 |
|--------|----------|
| text-only | 0.2678 |
| face-only | 0.1318 |
| majority-class | 0.0928 |
| fused: weighted-avg | 0.2662 |
| fused: confidence-gated | 0.2618 |
| fused: learned arbiter | 0.2693 |

**Finding:** naive fusion (weighted-avg, confidence-gated) does **not** beat
text-only — adding the weak face channel slightly degrades the strong text channel.
Only the **learned arbiter** avoids degradation, marginally exceeding text-only.

## RQ2 — conflict subset (2,025 / 2,610 = 77.6% where text-label != face-label)

| System | macro-F1 |
|--------|----------|
| text-only | 0.2561 |
| face-only | 0.0930 |
| majority-class | 0.0926 |
| fused: weighted-avg | 0.2551 |
| fused: confidence-gated | 0.2491 |
| fused: learned arbiter | 0.2630 |

**Finding:** when the channels disagree, the **learned arbiter is best** (0.263),
ahead of text-only (0.256) and clearly ahead of naive fusion. This is the core RQ2
evidence: a *learned* conflict policy beats naive averaging and beats text-alone.

## Implications
- Naive multimodal fusion can **hurt** when one channel is much weaker — argues for
  learned/gated arbitration rather than fixed averaging.
- The visual channel's value is limited here by (a) FER→TV domain shift and
  (b) ~11% frames with no detectable face.
- Naturalistic validation of the face channel belongs to the **user study (RQ3)**,
  where the webcam operates in-domain (real frontal faces), unlike MELD's TV frames.

## Reproduce
```
python evaluation/evaluate_meld.py \
  --text-model-dir models/weights/text \
  --face-model-path models/weights/face/face_net.pt
```
