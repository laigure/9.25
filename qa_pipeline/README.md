# 3EED spatial QA prototype

`generate_qa.py` reads one normalized scene per JSONL line and creates spatial
question-answer pairs from **GT box centers only**. It has no model dependency.

```powershell
python qa_pipeline/generate_qa.py qa_pipeline/example_scenes.jsonl qa_pipeline/example_qa.jsonl
```

## Metadata audit, run before the converter

`audit_3eed.py` runs on the dataset host (CPU only) and reports what the raw
`meta_info.json` actually contains: schema of `ground_info` and `others`,
`bbox_3d` length and per-index value ranges, `loader_class_match_rate` (how many
annotations survive the training loader's synonym match), spatial-word leakage,
same-frame multi-caption counts, and the ego/pose direction test.

Its decisive output is the **relation test**: every caption containing exactly
one of left/right/front/behind is turned into a vote for or against the four
candidate axis conventions, under each candidate frame interpretation. The
variant with the highest match rate is the frame the QA labels must be generated
in, so this decides both the converter's box handling and the QA frame.

```bash
cd /root/3eedqa/3EED
python audit_3eed.py --data-root data/3eed --split-dir data/3eed/splits \
    --splits val train --max-frames 0 --out /root/autodl-tmp/3eed_data/audit_report.json
```

Use `--max-frames 300` for a quick pass. `--synonym-file src/joint_det_dataset.py`
reads the synonym tables with `ast` so the loader's class match is reproduced
without importing torch. Variants compared per platform: `raw_metadata_frame`,
`rotated_by_sensor_view` (waymo), `pose_applied` and `pose_inverse_applied`
(any platform exposing a 4x4 `pose`).

## Raw 3EED converter

`convert_3eed.py` turns the raw `meta_info.json` tree into the scene JSONL
above. It was written against the measured schema and the measured frames
(`reports/audit_report.json`, `reports/projection_report.json`), not against
the loader's guesses:

- QA frame on all three platforms: `front = +x, left = +y, up = +z`.
- It mirrors the training loader exactly, because QA geometry must live in the
  frame the grounding model sees: waymo boxes are rotated about z by the
  sensor-view angle (`F 0, FL -45, FR 45, SL -90, SR 90`, yaw adjusted too),
  drone boxes shift `z += 1.8 m`, quad is untouched, and pose is never
  applied. `frame.points_transform` tells a downstream consumer what to apply
  to the raw cloud so points and boxes stay in one frame.
- `ground_info` entries become role `referred` objects with the caption as
  their referring expression; `others` entries become role `context` objects
  (no captions), deduplicated across a frame's entries by class and 1 cm
  center, and linked to a referred object when they coincide.
- No object id exists in the metadata: every id is generated and marked
  `id_source: generated`. `loader_visible` reproduces the loader's synonym
  class match (waymo 7464/7481; drone and quad complete).
- Layout measured: boxes are `[x, y, z, length, width, height, yaw, ...]`,
  first 7 used; drone/quad main boxes carry 9 values and unwrapped yaw, which
  is wrapped here. `bbox_2d_proj` is xyxy pixels.

```bash
cd /root/3eedqa/3EED
python /root/3eedqa/convert_3eed.py --data-root data/3eed --split-dir data/3eed/splits \
    --splits train val --synonym-file src/joint_det_dataset.py \
    --out /root/autodl-tmp/3eed_data/scenes/3eed_scenes.jsonl \
    --stats-out /root/autodl-tmp/3eed_data/scenes/3eed_scenes_stats.json
```

Full run: 20,367 scenes, 22,439 referred objects, 110,139 context objects in
6 s (97.8 MB JSONL). Two measured facts change how QA must be generated:

1. Captions almost always contain a relation word (waymo 7479/7481). The
   current "skip expressions containing spatial words" rule therefore yields
   **0 QA samples** on the real data. Leak control has to move to generating
   only relation types not mentioned in either caption, with the matched words
   recorded for audit.
2. Captions describe image-relative position ("left of center in the view"),
   and objects sit in a narrow forward view, so relation labels must use
   bearing-angle sectors, not Cartesian dominance. The prototype's
   dominant-axis rule only fits the synthetic example scenes.

## v1 QA pipeline (object bank -> pairs -> QA -> sanity check)

Four scripts build the first 3EED-grounded QA set. They run on the dataset
host (the token exports and the scene JSONL live there) and produce everything
under `qa_v1/`; the local `data/` folder is a copy of the artifacts. The
coordinate semantics used by every label are documented in
`coordinate_convention.md`; the token-selection mechanics (what the "annotated
span" is, and how to obtain tokens for two expressions) are analyzed in
`reports/token_selection_analysis.md`.

```bash
cd /root/3eedqa
python build_object_bank.py --out-dir /root/autodl-tmp/3eed_data/qa_v1
python build_pairs.py
python build_qa.py
python sanity_check.py
```

- `build_object_bank.py` joins every token NPZ row to its scene object by the
  verified key (meta_path minus `data/3eed/`, plus the cleaned caption; center
  difference measured < 0.0001 m). Writes `object_bank.jsonl` (9,694 records:
  waymo 3,782 / drone 2,984 / quad 2,928, each with GT box, predicted
  center/size, query index, match score, center error, IoU) and one stacked
  `tokens/{platform}_tokens.npz` `[N, 288]`; records point into it with
  `token_path` + `token_index`, tokens are never inlined in JSON.
- `build_pairs.py` makes same-frame **ordered** pairs (A relative to B) from
  waymo frames holding two or more tokens (903 frames; 2,600 pairs, cap
  `--max-pairs-per-frame 30`). All geometry comes from GT centers: dx/dy/dz,
  planar and 3D distance, bearing in degrees, a `bearing_sector` label
  (front |b|<=45, left 45-135, behind |b|>135, right -135..-45) and the
  independent-axis `relations` list that the answers speak. near/far labels
  are intentionally not produced; distances are only summarized.
- `build_qa.py` emits `qa_waymo.jsonl` in the target schema: both full and
  short expressions (3EED captions are full sentences; a heuristic noun-phrase
  cut keeps the question readable), token pointers, both GT boxes, structured
  geometry, relations, question, answer, and per-word leakage flags
  (`expression_collision` marks the 12 pairs whose short forms collide and
  which fall back to full captions). Answers use 3D center-to-center distance
  and are computed from GT boxes only.
- `sanity_check.py` prints 30 random pairs for eyeballing plus aggregate
  stats, and verifies by independent recomputation: same frame, distinct
  expressions, GT boxes equal to the bank, dx/dy/dz/dist recomputed from
  centers, direction labels consistent with signs, 288-D tokens loading from
  the npz, token row identity. Current run: 0 failures on the sample.

Measured facts that shape v1 (full numbers in `data/*_stats.json`):

- waymo val is the only place with multi-object frames (903 frames with >= 2
  tokens); drone and quad have exactly one captioned object per frame in the
  whole dataset, so their pairs must combine a referred and a context object.
- Answer direction words appear in the question for 2.9% of the 2,600 pairs
  after the short-expression cut (99.7% with full captions). The flag is
  audit metadata; the filtering policy is a downstream decision.
- Pairs come from denser frames: 41% of pairs have min grounding IoU < 0.25,
  versus an 80.7% IoU>=0.25 rate at the per-object level.

## v1 clean set, shared evaluation, downstream training

`build_qa_clean.py` turns `qa_waymo.jsonl` into `qa_waymo_clean.jsonl`: drops
the 75 leaked rows (all 12 collision rows are inside those 75), splits **by
scene** (seed 42, 70/15/15; no scene crosses a split; asserted), and adds
`qa_split` plus `grounding_quality` (target/reference IoU >= 0.5 ->
high_high / high_low / low_low, the 0.25 variant is only counted in stats).
Result: 2,525 rows over 888 scenes (train 1,816 / val 377 / test 332 pairs);
`leakage_check.verdict` is "pass" in `qa_waymo_clean_stats.json`. Provenance:
every v1 pair is a 3EED **val** scene (tokens were exported during validation
only), so this is a leakage-free scene-level split inside 3EED val, not the
3EED train split.

`qa_eval.py` is the shared answer parser and metric module every experiment
uses (format validity, left/right, front/behind, exact, distance MAE/median,
per-relation and per-quality-bucket breakdowns). `gt_baseline.py` is
Experiment A: it rebuilds the answers from GT geometry and reads them back
through the same parser (exact 1.0, format 1.0, distance MAE 0.025 m = the
one-decimal rounding bound) and also computes a random baseline (exact
0.2614, left/right 0.505, front/behind 0.5168, distance MAE 4.76 m).

`train_qa.py` trains and evaluates the downstream stack on the dataset host;
one script for all three modes so Experiments B and C stay byte-identical
(data, prompt, generation, evaluation):

```bash
python train_qa.py --mode projector --smoke --max-train 64
python train_qa.py --mode projector
python train_qa.py --mode eval --no-tokens            # text-only baseline
python train_qa.py --mode lora --init-projector runs/projector/best.pt
python train_qa.py --mode eval --init-projector runs/lora/best.pt --with-lora --eval-splits test
```

Prompt: system + question, two soft tokens at fixed markers ("Target object
token: ", "Reference object token: "), then the answer; labels cover the
answer text and its terminating `<|im_end|>` only. The projector is shared by
both tokens: `Linear(288,1024) -> GELU -> Linear(1024,hidden_size)`. LoRA
r=16, alpha=32, dropout=0.05, q/k/v/o_proj, bias=none. Qwen2.5-7B-Instruct is
bf16, frozen except the trained part; AdamW with projector_lr=lora_lr=1e-4,
weight_decay 0.01, 3 epochs, warmup_ratio 0.03, cosine, clip 1.0, per-device
batch 2 x grad_accum 4 (effective 8). Each run writes `history.json`,
`eval_<split>.json`, `samples20_val.json`, `samples_<split>.txt`,
`best.pt`/`last.pt` and `experiment_notes.md`.

`compare_runs.py` writes the B-vs-C report (metric table, delta exact
accuracy, train-loss drop, config-equality check, 20 identical validation
samples side by side) from the run directories; `run_experiments.sh` chains
smoke -> B -> text-only baseline -> C -> one final held-out test evaluation.

Results (2026-09-20, scene-level val n=377 / test n=332; full log in
`reports/experiment_log.md`, side-by-side in `reports/b_vs_c_report.md`):

| run | split | format | L/R | F/B | exact | buckets (hh/hl/ll) | dist MAE |
|---|---|---|---|---|---|---|---|
| B projector-only | val | 0.995 | 0.817 | 0.639 | 0.520 | 0.555/0.520/0.492 | 3.74 m |
| C + LoRA | val | 1.000 | 0.939 | 0.703 | 0.655 | 0.693/0.651/0.629 | 3.66 m |
| B projector-only | test | 1.000 | 0.804 | 0.627 | 0.530 | 0.624/0.553/0.380 | 4.23 m |
| C + LoRA | test | 1.000 | 0.901 | 0.654 | 0.599 | 0.692/0.650/0.413 | 4.01 m |

Baselines: a GT-rebuilt answer scores 1.0; random 0.2614; text-only (no
tokens, frozen LLM) 0.0. Attempt 1 of B was invalidated by a generation-prompt
bug (the generation context lacked the assistant prefix, so format validity
was 0.17); artifacts are archived under `runs/*_attempt1_bug` and the fix is
documented in `reports/experiment_log.md`. Remaining weak spots in both B and
C: the front/behind axis (0.65-0.70) and distance regression (predictions
cluster near ~10 m).

## Geometry ablation: D / E / oracle / relative-geometry

`train_qa.py` gained `--geom {none,center,box,relative}` (plus
`--pred-geometry` and the diagnostic-only `--oracle-center`): per-object
geometry from `tokens/waymo_geometry.npz` (built by `build_geom_npz.py` from
the exports, row-identity-checked against the token npz) is normalised with
**train-split-only** stats and fed alongside the token. `center`/`box` extend
the shared projector's first layer (288 -> 291/294) keeping all trained
columns and zero-initialising only the new ones, so the run continues from
C's `best.pt` with functional equivalence at init; `relative` keeps the token
projector and adds a separate geometry projector (5 -> 1024 -> hidden)
producing a third soft token from `[dx, dy, dz, d2d, d3d]` computed from
**predicted** centers. `box` is center+size (6-d) because the waymo grounding
model predicts axis-aligned boxes, so predicted yaw does not exist. All runs
share C's data/splits/prompt/eval; the Grounding model stays frozen
(precomputed exports).

Val (n=377) exact / F/B / distance MAE / Pearson:

| run | exact | F/B | dist MAE | Pearson |
|---|---|---|---|---|
| C baseline | 0.6552 | 0.7029 | 3.659 | 0.465 |
| D token+pred center | 0.7241 | 0.7692 | 3.487 | 0.522 |
| E token+pred box | 0.7268 | 0.7745 | 3.719 | 0.500 |
| oracle (GT center, diagnostic) | 0.7719 | 0.7984 | 3.610 | 0.505 |
| relative-geometry (diagnostic) | 0.8196 | 0.8541 | **2.538** | **0.722** |

Test (n=332): C 0.5994 / D 0.6837 / E 0.6717 / oracle 0.7048 / relgeo
0.7982 exact; relgeo distance MAE 2.436, Pearson 0.819. Conclusions: predicted
center fixes the front/behind axis (+6.6pp val, +6.6pp test); size adds
nothing; GT center does not fix distance (it is not a localization problem)
while relative geometry does, on both splits - the bottleneck is the LLM's
subtraction/distance step between two independent xyz tokens, not the 288-d
token. `ablation_report.py` regenerates the full tables plus distance
diagnostics (per-GT-distance-bin predicted means/MAE, Pearson, spread-ratio
collapse flag, 20 identical samples):

```bash
python qa_pipeline/ablation_report.py --run C=data/runs/lora \
    --run D=data/runs/d_center --run E=data/runs/e_box \
    --run oracle=data/runs/oracle_gt_center \
    --run relgeo=data/runs/relative_diag --baseline C \
    --out reports/ablation_c_d_e
```

Full detail: `reports/ablation_c_d_e.{md,json}` (val),
`reports/ablation_c_d_e_test.{md,json}` (test), and sections 6-13 of
`reports/experiment_log.md`.

## Scene contract

- `scene_id` identifies a frame, not just a sequence. `split` belongs to the
  scene so train/validation pairs cannot mix the same scene.
- `point_cloud_path` points to the raw cloud. `platform` identifies waymo,
  drone, or quad. RGB paths and original metadata paths can be added.
- `frame.name`, `frame.units`, `frame.right_axis`, `frame.forward_axis`, and
  `frame.up_axis`
  define the viewpoint used for relations. All GT boxes in a scene must use
  this same frame. The axes are unit vectors and measured in meters.
- Each object has a stable `object_id`, `category`, `gt_box.center`,
  `gt_box.size`, `gt_box.yaw_rad`, and `referring_expressions`. Preserve the
  original annotation ID in `object_id` where available. A generated ID must
  be marked as generated in a future converter.
- `gt_box` is a GT annotation. Model predictions and query features belong in
  a separate inference record and must never overwrite this field.

The prototype uses a dominant horizontal axis to label left, right, front,
or behind and skips diagonal or near-coincident pairs. `distance_m` is the
bird's-eye-view center-to-center distance. `near` and `far` need an explicit
reference population and threshold policy before labels can be generated.
Expressions containing obvious spatial relation words are skipped so the
question does not reveal its own answer. This rule is intentionally
conservative and requires review for real 3EED captions.

## Grounder token contract

The official model has 256 candidate queries with `d_model=288` by default.
Calling `model(batch, return_query_features=True)` now returns
`end_points["last_query_features"]` with shape `[B, 256, 288]`. Each candidate
also has `last_center`, `last_pred_size`, and a 64-D `last_proj_queries`
matching vector. Select a candidate using the same language-span scoring
policy as the evaluator, and record its query index, score, predicted box,
and source checkpoint. One GT object does **not** have a permanent query index;
multiple candidates may overlap it. For training, distinguish predicted
top-1 selection from oracle IoU matching. The official evaluator's query
ranking uses a target text span built from the annotation. The exporter
therefore provides an **annotated-span** result, which is useful for research
but is not yet a deployable free-form grounding result. A later inference
path must predict the text span or rank queries without that annotation.

For a target-reference pair, run or batch two expressions against the same
scene and cache each selected token by `(scene_id, expression_id)`. These
tokens are conditioned on their respective captions. Relation words inside
the captions can leak the QA answer into the token, so evaluation needs
relation-neutral expressions or a separate caption leakage audit.

Add `--export_qa_tokens <output_dir>` to the repository's validation command.
Each rank writes one `.npz` per batch containing the selected 288-D token,
query index, match score, predicted box, GT box, caption, and metadata path.
This path has been exercised on a real 3EED Waymo sample with the public
checkpoint on the remote RTX 4090 host. See `../RESEARCH_PROGRESS.md` for
locations, the full validation run, and remaining checks.

## Remaining integration checks

1. ~~Run `audit_3eed.py`, then write the converter~~ **Done**: frames are
   measured, the converter has run on all 20,367 frames, and the local report
   copies are in `reports/`. Only `ground_info` provides captions; `others`
   becomes role `context`. Waymo has 1,753 frames with two or more captioned
   objects; drone and quad have exactly one caption per frame.
2. ~~Apply the winning relation-test variant to all boxes~~ **Done** in
   `convert_3eed.py` (waymo view rotation, drone z offset, no pose), declared
   per scene via `frame.points_transform`. Next: extend QA generation with
   bearing-angle labels, caption leak control, and a caption-free
   geometry-only sample tier.
3. ~~Complete full validation, record metrics and token export counts, and
   compare query selection with the official evaluator~~ **Done**: all three
   platforms evaluated and exported (1,212 npz / 9,694 tokens); per-row IoU
   recomputation reproduces the official Acc@0.25/0.5 exactly, so the exported
   rows correspond one-to-one with the evaluator's given-span metric.
4. Train and evaluate GT geometry QA first. Then freeze the grounder,
   substitute exported 288-D features and predicted boxes, and measure the
   grounding-induced performance drop with the QA labels fixed. Include
   geometry-only, token-only, and token-plus-geometry ablations to establish
   whether the projector and LLM use the learned representation.
