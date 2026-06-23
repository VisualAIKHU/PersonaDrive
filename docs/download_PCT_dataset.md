# Preparation of PCT Dataset

The **Persona-Conditioned Trajectory (PCT)** dataset is a layer of persona
annotations on top of the public [OpenScene-v1.1](https://github.com/OpenDriveLab/OpenScene)
/ [NAVSIM](https://github.com/autonomousvision/navsim) scenes. PCT ships **no
sensors**: it is a set of per-scene JSON files keyed by the OpenScene scene token,
joined back to the sensor data purely by filename.

Each scene is annotated for the **9 personas** that span
**Temporal Urgency** (High / Mid / Low) × **Ride Comfort** (Low / Mid / High):

```
UH_CL  UH_CM  UH_CH
UM_CL  UM_CM  UM_CH      (UM_CM = neutral / default persona)
UL_CL  UL_CM  UL_CH
```

Every persona entry holds a natural-language `user_intent` utterance, a future
`trajectory` (10 × `[x, y, heading]` waypoints), and the persona `params`. Preparing
PCT for training/evaluation is a three-step process:

1. build the standard NAVSIM caches,
2. download and unpack the PCT annotations,
3. inject the persona annotations into the caches.

## 0. Prerequisites

Complete [Preparation of PersonaDrive environment](setup_env.md) first, so that
`$NAVSIM_EXP_ROOT` is exported and the NAVSIM dataset / maps are in place.

## 1. Build the base NAVSIM caches

The persona annotations are injected into the same `training_cache/` and
`metric_cache/` trees that the standard NAVSIM caching scripts produce. Build them
first:

```bash
# cache dataset for training
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
        agent=personadrive_agent \
        experiment_name=training_personadrive_agent \
        train_test_split=navtrain

# cache dataset for evaluation
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
        train_test_split=navtest \
        cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache
```

This produces:

```
$NAVSIM_EXP_ROOT
├── training_cache/<log>/<token>/transfuser_feature.gz
│                                transfuser_target.gz
└── metric_cache/<log>/<token>/metric_cache.pkl
```

## 2. Download and unpack the PCT annotations

Download the PCT dataset archive from
[PCT Dataset (Google Drive)](https://drive.google.com/file/d/17NUcrvEfkmmEjQF1D0mtBMHCYsjqbsec/view?usp=drive_link).
The instructions below assume you have placed `PersonaDrive_dataset.tar.gz` at the
repository root. Unpack it under `$NAVSIM_EXP_ROOT`:

```bash
mkdir -p $NAVSIM_EXP_ROOT/personadrive
tar -xzf PersonaDrive_dataset.tar.gz -C $NAVSIM_EXP_ROOT/personadrive
# → $NAVSIM_EXP_ROOT/personadrive/per_scene/<token>.json   (115,434 files)
```

## 3. Inject persona annotations into the caches

The injection scripts live in [`tools/personadrive_cache`](../tools/personadrive_cache)
and layer the 9 persona annotations on top of the caches you built in step 1. See
the [tool README](../tools/personadrive_cache/README.md) for full details and flags.

### 3.1 Training cache

For every token in `training_cache/<log>/<token>/`, this fills:

- `transfuser_feature.gz` ← `input_ids` (9 × `[max_len]`), `attention_mask` (9 × `[max_len]`)
- `transfuser_target.gz` ← `trajectories` (9 × `[8, 3]`), `categories` (9 × one-hot `[9]`)

```bash
cd tools/personadrive_cache
python inject_to_train_cache.py \
    --cache_dir     $NAVSIM_EXP_ROOT/training_cache \
    --per_scene_dir $NAVSIM_EXP_ROOT/personadrive/per_scene \
    --num_workers 8
# add --dry_run first to validate without writing
```

> Tokens lacking a matching `per_scene/<token>.json` are skipped — restrict your
> training split to the injected tokens.

### 3.2 Metric cache (optional)

The PDM scorer state in `metric_cache.pkl` needs nothing added, and the main scorer
`run_pdm_score.py` reads the persona annotations from `per_scene_dir` directly (see
[Training and Evaluation](train_eval.md)). **You can skip this step for standard
training/evaluation.** It only writes a `gpt.json` next to each `metric_cache.pkl`
for older / external tooling that expects the annotations embedded in the metric
cache:

```bash
python inject_to_metric_cache.py \
    --metric_cache_dir $NAVSIM_EXP_ROOT/metric_cache \
    --per_scene_dir    $NAVSIM_EXP_ROOT/personadrive/per_scene \
    --tokenize --num_workers 8
```

`--tokenize` additionally embeds pre-computed `input_ids` / `attention_mask` per
persona (keyed by backbone) so the scorer fast-path can skip on-the-fly tokenisation.

## Notes

- Default text backbone: `sentence-transformers/all-MiniLM-L6-v2`, `max_length=100`.
  Re-run with `--bert_backbone ... --max_length ...` to switch encoders (keep this in
  sync with `bert_backbone` in the agent config).
- Both scripts are idempotent — re-running overwrites the injected keys/files.
- The training target uses the first **8** of the 10 JSON waypoints, `[x, y, heading]`.

## Next steps

- [Training and Evaluation](train_eval.md)
