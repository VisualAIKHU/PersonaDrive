# PersonaDrive (PCT) cache injection

Expand a plain **OpenScene / NavSim** cache into the **PersonaDrive (PCT)**
persona-conditioned dataset. These scripts assume you have *only* downloaded
OpenScene/NavSim and unpacked the PersonaDrive annotations — they layer the 9
persona annotations on top of the caches you already built.

PersonaDrive ships no sensors. It is a set of per-scene JSON annotations keyed by
the OpenScene scene token, joined back to the sensor data purely by filename. The
9 personas span Urgency (High/Mid/Low) × Comfort (Low/Mid/High):
`UH_CL UH_CM UH_CH UM_CL UM_CM UM_CH UL_CL UL_CM UL_CH` (`UM_CM` = neutral).

## 0. Prerequisites

Download OpenScene/NavSim and build the standard caches per the official NavSim
instructions, which export `NAVSIM_EXP_ROOT` and create the `training_cache/` and
`metric_cache/` trees. Then unpack the annotations:

```bash
mkdir -p $NAVSIM_EXP_ROOT/personadrive
tar -xzf PersonaDrive_dataset.tar.gz -C $NAVSIM_EXP_ROOT/personadrive
# → $NAVSIM_EXP_ROOT/personadrive/per_scene/<token>.json   (115,434 files)
```

## Files

| file | purpose |
| --- | --- |
| `personadrive_common.py`     | shared constants, JSON loading, tokenisation, gz IO |
| `inject_to_train_cache.py`   | **training**: fill feature/target `.gz` cache files |
| `inject_to_metric_cache.py`  | **evaluation**: write per-token `gpt.json` |

## 1. Training cache

For every token in `training_cache/<log>/<token>/`:

* `transfuser_feature.gz` ← `input_ids` (9×`[max_len]`), `attention_mask` (9×`[max_len]`)
* `transfuser_target.gz` ← `trajectories` (9×`[8,3]`), `categories` (9× one-hot `[9]`)

The agent reads text tokens from the feature file and the per-persona targets from
the standard `transfuser_target.gz` (the persona keys are added in-place, alongside
the existing GT trajectory and BEV labels).

```bash
python inject_to_train_cache.py \
    --cache_dir     $NAVSIM_EXP_ROOT/training_cache \
    --per_scene_dir $NAVSIM_EXP_ROOT/personadrive/per_scene \
    --num_workers 8
# add --dry_run first to validate without writing
```

Tokens lacking a `per_scene/<token>.json` are skipped — restrict your training
split to the injected tokens.

## 2. Metric cache

The PDM scorer state in `metric_cache.pkl` needs nothing added. This script writes
a `gpt.json` (the persona annotations) next to each `metric_cache.pkl`, which the
visualisation / legacy eval paths read. The main scorer (`run_pdm_score.py`) can
instead read `per_scene_dir` directly, so this step is only needed for those paths.

```bash
python inject_to_metric_cache.py \
    --metric_cache_dir $NAVSIM_EXP_ROOT/metric_cache \
    --per_scene_dir    $NAVSIM_EXP_ROOT/personadrive/per_scene \
    --tokenize --num_workers 8
```

`--tokenize` additionally embeds pre-computed `input_ids`/`attention_mask` per
persona (keyed by backbone) so the scorer fast-path skips on-the-fly tokenisation.

## Notes

* Default text backbone: `sentence-transformers/all-MiniLM-L6-v2`, `max_length=100`.
  Re-run with `--bert_backbone ... --max_length ...` to switch encoders.
* Both scripts are idempotent — re-running overwrites the injected keys/files.
* Trajectory target = first 8 of the 10 JSON waypoints, `[x, y, heading]`.
