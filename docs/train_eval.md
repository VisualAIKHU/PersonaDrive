# PersonaDrive Training and Evaluation

This guide assumes you have already prepared the environment
([setup_env.md](setup_env.md)) and the PCT-injected caches
([download_PCT_dataset.md](download_PCT_dataset.md)).

## 0. Before you start

If your training machine does not have network access, download the pretrained
ResNet-34 backbone from [huggingface/timm](https://huggingface.co/timm/resnet34.a1_in1k)
and the clustered anchors
[`kmeans_navsim_traj_20.npy`](https://github.com/hustvl/DiffusionDrive/releases/download/DiffusionDrive_88p1_PDMS_Eval_file/kmeans_navsim_traj_20.npy),
then upload them to the machine.

Set the following in `navsim/agents/personadrive/transfuser_config.py`:

- `bkb_path` → path to the downloaded pretrained ResNet-34 weights
- `plan_anchor_path` → path to the downloaded `kmeans_navsim_traj_20.npy`
- `bert_backbone` → text encoder (default `sentence-transformers/all-MiniLM-L6-v2`);
  point it to a local path if the machine is offline

## 1. Training

PersonaDrive trains on the persona-injected training cache. The agent reads the text
tokens from `transfuser_feature.gz` and the per-persona targets from the
`transfuser_target.gz` files produced in
[Preparation of PCT Dataset](download_PCT_dataset.md#31-training-cache).

```bash
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
        agent=personadrive_agent \
        experiment_name=training_personadrive_agent \
        train_test_split=navtrain \
        split=trainval \
        trainer.params.max_epochs=100 \
        cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
        use_cache_without_dataset=True \
        force_cache_computation=False
```

PersonaDrive's persona-conditioning components — Persona-Conditioned Anchor Transform
(PCAT), Persona-Conditioned Multi-Modal Fusion (PCMF), the Hierarchical Guide Loss, and
the Axis-Decomposed Diversity Loss — are always enabled. Their hyperparameters
(e.g. `guide_weight`, `diversity_cross_weight`, `diversity_tau`, `neutral_persona_idx`)
can be tuned in `navsim/planning/script/config/common/agent/personadrive_agent.yaml`.

## 2. Evaluation

Set `CKPT` to your trained checkpoint, then run the PDM scorer. You can also evaluate
our released checkpoint — download it from
[Checkpoint (Google Drive)](https://drive.google.com/file/d/1Aq_xFWfVGAi6W8m7NCnV6ZcWJzqWNPWz/view?usp=drive_link):

```bash
export CKPT=/path/to/your/checkpoint.pth

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
        train_test_split=navtest \
        agent=personadrive_agent \
        worker=ray_distributed \
        agent.checkpoint_path=$CKPT \
        metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache \
        per_scene_dir=$NAVSIM_EXP_ROOT/personadrive/per_scene \
        driving_purpose=all \
        experiment_name=personadrive_agent_eval
```

- `driving_purpose=all` evaluates all 9 personas and reports the averaged metrics in
  the [README](../README.md#results). To evaluate a single persona instead, set it to
  one of `UH_CL UH_CM UH_CH UM_CL UM_CM UM_CH UL_CL UL_CM UL_CH`.
- `per_scene_dir` points the scorer at the PCT annotations so it can read each scene's
  persona `user_intent`. It defaults to `$NAVSIM_EXP_ROOT/personadrive/per_scene` (where
  [Preparation of PCT Dataset](download_PCT_dataset.md#2-download-and-unpack-the-pct-annotations)
  unpacks the dataset), so the override above is only needed if you placed it elsewhere.
- `metric_cache_path` must point at the metric cache built in
  [Preparation of PCT Dataset](download_PCT_dataset.md#1-build-the-base-navsim-caches).
- The run reports the per-persona and averaged ADE / FDE / PDMS metrics (cf. the
  Results table in the [README](../README.md#results)).

### Persona grid visualization

To save the 3×3 persona-grid qualitative visualizations, add the `vis=true` flag:

```bash
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
        train_test_split=navtest \
        agent=personadrive_agent \
        worker=ray_distributed \
        agent.checkpoint_path=$CKPT \
        metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache \
        per_scene_dir=$NAVSIM_EXP_ROOT/personadrive/per_scene \
        driving_purpose=all \
        vis=true \
        experiment_name=personadrive_agent_eval
```

See [`tutorial/visualization_PersonaDrive.ipynb`](../tutorial/visualization_PersonaDrive.ipynb)
for an interactive walkthrough of the qualitative results.
