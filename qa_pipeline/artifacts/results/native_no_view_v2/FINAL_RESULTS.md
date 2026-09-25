# Native no-view QA final results

The public prompt contains natural non-spatial descriptions and implicit Grounding tokens. View names and explicit geometry are absent.

## Main metrics

| Run | Strict | Grounding conditional | End-to-end | Inverse consistency |
|---|---:|---:|---:|---:|
| noaux_projector | 47.62% | 47.00% | 26.33% | 13.82% |
| noaux_lora | 50.05% | 49.37% | 27.65% | 29.30% |
| noaux_shuffled | 49.60% | 48.29% | 27.05% | 30.16% |
| aux_projector | 51.83% | 53.26% | 29.84% | 31.55% |
| aux_lora | 57.93% | 61.70% | 34.57% | 55.56% |
| aux_shuffled | 47.02% | 45.44% | 25.45% | 30.09% |

## Token ablation

- **noaux:** correct minus shuffled = 0.45 pp; paired 95% CI [-0.05, 0.94] pp; correct-only/shuffled-only = 493/439.
- **aux:** correct minus shuffled = 10.90 pp; paired 95% CI [10.05, 11.76] pp; correct-only/shuffled-only = 2131/812.

## Auxiliary LoRA by scenario

| Scenario | Correct token | Shuffled token | Gain |
|---|---:|---:|---:|
| cardinal_relation | 57.28% | 31.48% | +25.79 pp |
| ego_absolute_distance | 0.84% | 0.14% | +0.70 pp |
| ego_motion_reasoning | 82.90% | 81.88% | +1.03 pp |
| ego_relative_distance | 75.33% | 54.91% | +20.42 pp |
| object_absolute_distance | 0.92% | 0.66% | +0.26 pp |
| object_motion_reasoning | 70.21% | 59.03% | +11.18 pp |
| object_relative_distance | 63.04% | 58.70% | +4.35 pp |

## Interpretation

- Without auxiliary relation supervision, correct tokens do not significantly outperform shuffled tokens.
- Auxiliary relation supervision produces a large, statistically clear token-dependent gain, especially for cardinal relations, relative distance and object motion.
- Exact metric-distance generation remains near zero and needs a separate design; the overall distance category must not hide this.
- Ego-motion remains mostly answerable from language priors because its correct/shuffled gap is small.
