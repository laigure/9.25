# Native no-view QA detailed results

Validation contains 12,096 questions. A prediction passes only when the full
canonical sentence, one-sentence format and private subject-predicate-object or
numeric key all pass. Coordinates and view names are absent from model input.

## Meaning of the columns

- **Noaux LoRA:** language-answer supervision only.
- **Aux Projector / Aux LoRA:** training additionally uses private GT geometry
  to supervise pairwise left/right and front/behind labels. Geometry does not
  enter the token, prompt or Qwen embedding.
- **Shuffled:** the same aux LoRA receives same-class tokens from other objects.
- **Token gain:** Aux LoRA minus Shuffled.
- **Grounding coverage:** all objects required by the question have predicted
  boxes with IoU >= 0.25.
- **Conditional:** Aux LoRA accuracy restricted to Grounding-correct questions.
- **End-to-end:** both Grounding and the strict QA answer are correct.

## Every question type

| Question type | N | Noaux LoRA | Aux Projector | Aux LoRA | Shuffled | Token gain | Grounding coverage | Conditional | End-to-end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cardinal relation | 3,024 | 37.83% | 46.89% | **57.28%** | 31.48% | **+25.79 pp** | 56.61% | 67.70% | 38.33% |
| Ego absolute distance | 1,421 | 0.42% | 0.63% | **0.84%** | 0.14% | +0.70 pp | 56.58% | 1.00% | 0.56% |
| Object absolute distance | 757 | 0.53% | 0.40% | **0.92%** | 0.66% | +0.26 pp | 55.48% | 0.48% | 0.26% |
| Ego relative distance | 754 | 63.79% | 69.89% | **75.33%** | 54.91% | **+20.42 pp** | 55.84% | 81.24% | 45.36% |
| Object relative distance | 92 | 61.96% | 44.57% | 63.04% | 58.70% | +4.35 pp | 38.04% | 62.86% | 23.91% |
| Ego motion | 3,024 | 82.90% | 80.62% | 82.90% | 81.88% | +1.03 pp | 55.36% | 83.99% | 46.49% |
| Object motion | 3,024 | 61.34% | 60.62% | **70.21%** | 59.03% | **+11.18 pp** | 56.55% | 72.63% | 41.07% |

The aggregate distance score is 21.33%, but it combines strong relative-distance
classification with failed exact-meter generation. It must not be reported
without the three separate distance rows above.

## Cardinal relation labels

| Relation | N | Noaux LoRA | Aux LoRA | Shuffled | Token gain | Conditional | End-to-end |
|---|---:|---:|---:|---:|---:|---:|---:|
| Left | 574 | 16.38% | **52.44%** | 11.67% | **+40.77 pp** | 61.29% | 39.72% |
| Right | 574 | 31.01% | **51.74%** | 11.67% | **+40.07 pp** | 61.02% | 39.55% |
| Front | 938 | 40.62% | **60.77%** | 43.50% | **+17.27 pp** | 72.93% | 37.63% |
| Behind | 938 | 52.35% | **60.13%** | 43.71% | **+16.42 pp** | 72.52% | 37.42% |

The left/right results are the clearest evidence that auxiliary supervision
extracts spatial information from the 288-dimensional tokens. Front/behind has
a higher shuffled baseline, so its raw accuracy contains more label prior.

## Motion direction

| Task and direction | Aux LoRA | Shuffled | Token gain |
|---|---:|---:|---:|
| Ego forward | 99.74% | 99.74% | 0.00 pp |
| Ego backward | 99.87% | 99.87% | 0.00 pp |
| Ego left | 65.54% | 64.72% | +0.81 pp |
| Ego right | 65.83% | 62.55% | +3.29 pp |
| Object forward | **75.75%** | 58.11% | **+17.64 pp** |
| Object backward | **77.43%** | 57.33% | **+20.10 pp** |
| Object left | 63.85% | 62.66% | +1.19 pp |
| Object right | 63.26% | 58.01% | +5.25 pp |

Ego forward/backward is almost deterministic from the dataset and does not test
token reasoning. For object motion, the noaux model gets only 0.60% of the
`gets closer` cases while scoring 99.30% on `does not get closer`; it mainly
predicts the majority outcome. Auxiliary supervision raises `gets closer` to
42.39%, demonstrating real improvement despite the remaining imbalance.

## Exact metric-distance diagnostics

Every answer contains one parsable number, so the failure is numerical accuracy
rather than malformed language.

| Task | Exact 0.05 m | Within 0.5 m | Within 1 m | Within 2 m | Within 5 m | MAE | Shuffled MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ego absolute distance | 0.84% | 10.34% | 17.59% | 34.20% | 69.46% | 4.33 m | 9.44 m |
| Object absolute distance | 0.92% | 10.04% | 22.32% | 38.18% | 70.81% | 4.03 m | 5.32 m |

The tokens contain coarse metric information, especially for ego distance, but
the current language-generation objective cannot recover a one-decimal distance
reliably. A separate regression head or distance bins should be evaluated.

## Two versus three targets

| Target count | N | Aux LoRA | Shuffled | Token gain | Grounding coverage | Conditional | End-to-end |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 10,624 | 57.93% | 46.60% | **+11.33 pp** | 58.41% | 61.99% | 36.21% |
| 3 | 1,472 | 57.88% | 50.07% | **+7.81 pp** | 38.72% | 58.60% | 22.69% |

The QA backend performs similarly for two and three targets. The three-target
end-to-end drop is mainly caused by Grounding coverage (38.72% versus 58.41%),
not by a collapse of the QA backend.

## Statistical interpretation

- Overall noaux token gain: +0.45 pp, paired 95% CI [-0.05, 0.94]; not clear.
- Overall aux token gain: +10.90 pp, paired 95% CI [10.05, 11.76]; clear.
- Cardinal relation gain: +25.79 pp, 95% CI [23.44, 28.15].
- Ego relative-distance gain: +20.42 pp, 95% CI [16.01, 24.84].
- Object-motion gain: +11.18 pp, 95% CI [9.33, 13.02].
- Object-relative-distance has only 92 questions and its interval includes zero;
  the apparent +4.35 pp is not reliable yet.

The evidence supports a narrow conclusion: auxiliary relation supervision makes
the frozen Grounding tokens useful for direction, relative distance and some
object-motion reasoning. It does not yet establish reliable exact metric
distance or unbiased ego-motion reasoning.
