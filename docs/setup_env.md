# Preparation of PersonaDrive environment

PersonaDrive is built on top of the [NAVSIM](https://github.com/autonomousvision/navsim)
devkit. This guide sets up the Python environment and the extra dependencies that
PersonaDrive needs on top of a working NAVSIM installation.

> **Before you start**, make sure you have already prepared the NAVSIM dataset and
> environment variables by following
> [Getting started from NAVSIM environment preparation](https://github.com/autonomousvision/navsim?tab=readme-ov-file#getting-started-).
> In particular, the following variables must be exported (e.g. in your `~/.bashrc`):
>
> ```bash
> export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
> export NUPLAN_MAPS_ROOT="$HOME/navsim_workspace/dataset/maps"
> export NAVSIM_EXP_ROOT="$HOME/navsim_workspace/exp"
> export NAVSIM_DEVKIT_ROOT="$HOME/navsim_workspace/navsim"
> export OPENSCENE_DATA_ROOT="$HOME/navsim_workspace/dataset"
> ```

## 1. Clone the repository

```bash
git clone https://github.com/VisualAIKHU/PersonaDrive.git
cd PersonaDrive
```

## 2. Create the conda environment

We provide an `environment.yml` (Python 3.9) that installs NAVSIM and all of its
dependencies from `requirements.txt`:

```bash
conda env create --name navsim -f environment.yml
conda activate navsim
pip install -e .
```

## 3. Install PersonaDrive-specific dependencies

PersonaDrive conditions trajectories on natural-language persona descriptions, so it
relies on a HuggingFace text encoder (default:
`sentence-transformers/all-MiniLM-L6-v2`). Install the `transformers` library, which
is not part of the base NAVSIM requirements:

```bash
pip install transformers
```

The text backbone is downloaded automatically from the HuggingFace Hub on first use.
If your training/evaluation machine has no network access, pre-download the encoder
and point `bert_backbone` (in
`navsim/agents/personadrive/transfuser_config.py` and the agent config) to the local
path.

## 4. (Optional) Pre-download model weights for offline machines

For offline training you will also need:

- the pretrained ResNet-34 backbone from
  [huggingface/timm](https://huggingface.co/timm/resnet34.a1_in1k), and
- the clustered anchors
  [`kmeans_navsim_traj_20.npy`](https://github.com/hustvl/DiffusionDrive/releases/download/DiffusionDrive_88p1_PDMS_Eval_file/kmeans_navsim_traj_20.npy).

Set `bkb_path` and `plan_anchor_path` in
`navsim/agents/personadrive/transfuser_config.py` to the downloaded files. See
[Training and Evaluation](train_eval.md) for details.

## Next steps

- [Preparation of PCT Dataset](download_PCT_dataset.md)
- [Training and Evaluation](train_eval.md)
