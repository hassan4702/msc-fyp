# MELD Evaluation Results

Test split = 2,610 utterances. Arbiter trained on 9,989 train utterances.

Four pipelines are reported, because the differences between them are results in their own
right:

- **(A)** Haar cascade + from-scratch CNN, with the crop-framing bug (run 2026-06-09)
- **(B)** the same models, crop framing corrected
- **(C)** MediaPipe BlazeFace + ImageNet-pretrained ResNet-18 — **the shipped pipeline**
- **(D)** (C) additionally fine-tuned on MELD train (`training/finetune_meld.py`)

**Critical framing — (A)–(C) are cross-corpus transfer, not in-domain.** The text model was
fine-tuned on **GoEmotions**, the face model on **FER-2013**, then both applied to **MELD with no
MELD fine-tuning**. Those numbers measure *generalization* to a new domain (TV dialogue), which is
why absolute macro-F1 is ~0.27, well below in-domain MELD text SOTA. (D) closes that gap for the
face channel only.

## RQ1 — macro-F1 on MELD test (all utterances)

| System | (A) original | (B) crop fixed | (C) ResNet-18 | (D) + MELD-tuned |
|--------|--------------|----------------|---------------|------------------|
| text-only | 0.2678 | 0.2678 | 0.2678 | 0.2678 |
| face-only | 0.1318 | 0.1398 | 0.1210 | **0.1506** |
| majority-class | 0.0928 | 0.0928 | 0.0928 | 0.0928 |
| fused: weighted-avg | 0.2662 | 0.2655 | 0.2505 | 0.2547 |
| fused: confidence-gated | 0.2618 | 0.2640 | 0.2451 | 0.2586 |
| fused: learned arbiter *(train)* | 0.2693 | **0.2751** | 0.2679 | 0.1954 † |
| fused: learned arbiter *(dev)* | — | — | 0.2323 | 0.2314 |
| *face-detection coverage* | *88.7%* | *88.7%* | *96.6%* | *96.6%* |

## RQ2 — conflict subset (text label != face label)

| System | (A) original | (B) crop fixed | (C) ResNet-18 | (D) + MELD-tuned |
|--------|--------------|----------------|---------------|------------------|
| text-only | 0.2561 | 0.2567 | 0.2591 | 0.2691 |
| fused: learned arbiter *(train)* | 0.2630 | **0.2648** | 0.2574 | 0.1754 † |
| fused: weighted-avg | 0.2551 | 0.2542 | 0.2425 | 0.2535 |
| fused: confidence-gated | 0.2491 | 0.2520 | 0.2359 | 0.2571 |
| face-only | 0.0930 | 0.1006 | 0.0930 | 0.1173 |
| *subset size* | *2,025* | *1,969* | *2,167* | *2,090* |

**Text-only is identical to four decimal places across all four runs** — the control confirming
that only the face path changed between them.

## † The arbiter cannot train on a split its face model was fine-tuned on

The (D) figures marked † are **invalid**, and are kept only to document the trap. The arbiter
trains on MELD *train*; in (D) the face model was itself fine-tuned on MELD *train*, so the
arbiter learns from face predictions on utterances the face model memorised (training loss fell
to 0.89), over-trusts the face channel, and collapses on test — 0.2679 → 0.1954. (A)–(C) are
immune because those face models never saw MELD, which is what makes the bug easy to miss the
moment in-domain fine-tuning is added. Use `--arbiter-split dev`.

That fix yields 0.2314, still below (C)'s 0.2679 — leak, or just 9× less arbiter data (1,109 vs
9,989 records)? Running (C) under the same dev arbiter isolates it:

| arbiter training split | (C) | (D) |
|---|---|---|
| train — 9,989 records | 0.2679 | 0.1954 † leaked |
| dev — 1,109 records | 0.2323 | 0.2314 |

Matched, they are **identical within noise**. The drop was arbiter data starvation, not the
fine-tuned face model. Secondary lesson: the learned arbiter is **data-hungry**, losing 3.5pp when
its training set shrinks 9×.

## Finding 1 — a preprocessing-geometry bug was worth ~6pp

(A) → (B) changes nothing but the crop. Inference had been feeding the model the raw Haar
bounding box, ~0.807× the FER-2013 framing the model was trained on. Measured in-domain on 3,589
held-out FER images, that framing mismatch alone cost **6.0pp accuracy / 7.5pp macro-F1**. On
MELD it moved face-only from 0.1318 to 0.1398 and the learned arbiter from 0.2693 to 0.2751. The
conflict subset also shrank (2,025 → 1,969), i.e. the corrected face channel agrees with the text
channel more often — independent evidence the fix moved predictions toward the truth.

## Finding 2 — in-domain gains did not transfer

Pipeline (C) is **+11.1pp macro-F1 on FER-2013** (0.540 → 0.651) and detects 8pp more faces, yet
is **worse on MELD** (face-only 0.1398 → 0.1210).

The obvious explanation — that Haar's 11% missed faces were scored as "neutral", and MELD is 48%
neutral, so the old pipeline got free correct answers — was tested and **refuted**. Restricted to
the 2,242 utterances where *both* pipelines detect a face, so neither receives a free "neutral",
the old model still leads **0.1305 to 0.1128**.

The honest reading: the larger pretrained model fits FER-2013's distribution (posed, frontal,
48×48 grayscale stills) more tightly and therefore generalises *less* well to MELD's (TV frames,
motion blur, profiles, cinematic lighting). Higher in-domain accuracy bought worse cross-corpus
transfer.

**Pipeline (D) confirms that diagnosis by reversing it.** Fine-tuning the same ResNet-18 on MELD
train lifts face-only from 0.1210 to **0.1506**, the best face channel here. So (C)'s regression
was domain shift, not a defect in the backbone.

## Finding 3 — a better face channel did not produce better fusion

Face-only improved **25% relative** (0.1210 → 0.1506) between (C) and (D). Fusion under a matched
arbiter moved 0.2323 → 0.2314 — nothing. And no pipeline's fusion beat text-only (0.2678).

## Implications

- Naive multimodal fusion can **hurt** when one channel is much weaker — this holds across all
  four pipelines, so it is a property of the fusion design, not of any one face model. It argues
  for learned/gated arbitration rather than fixed averaging.
- **The binding constraint is face-channel quality — and improving it was not enough.** Face-only
  sits at 0.12–0.15 against a 0.093 majority baseline and ~0.14 chance for 7 classes. A 25%
  relative gain moved fusion by 0.001. No fusion policy extracts much from a channel this close
  to chance.
- **MELD's labels are the real ceiling.** They are utterance-level and were annotated from audio +
  text + video together, so a sampled frame frequently does not show the labelled emotion. During
  fine-tuning, training loss fell 2.396 → 0.891 while dev macro-F1 stalled near 0.14 — the model
  fitting noise rather than signal.
- Naturalistic validation of the face channel therefore belongs to the **user study (RQ3)**, where
  the webcam operates in-domain (large, frontal, well-lit faces) — much closer to FER-2013's
  setting than to MELD's, and the only place the deployment domain is actually measured.

## Reproducing

```bash
python evaluation/evaluate_meld.py --face-model-path models/weights/face/resnet18.pt
python evaluation/evaluate_meld.py --face-model-path models/weights/face/resnet18_meld.pt \
    --arbiter-split dev      # --arbiter-split dev is REQUIRED for any MELD-fine-tuned model
```
