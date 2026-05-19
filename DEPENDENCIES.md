# RoboPianist-224R Dependency Guide

This document explains the dependency requirements for running the RoboPianist training code, particularly the version constraints needed to avoid compatibility issues.

## The Problem

The `robopianist-rl` codebase uses older JAX APIs that have been deprecated or removed in newer versions. Additionally, the JAX ecosystem (JAX, NumPy, SciPy, Flax, Optax) has tight version coupling where mismatched versions cause cryptic errors.

## Known Compatibility Issues

| Error | Cause | Fix |
|-------|-------|-----|
| `AttributeError: module 'jax.random' has no attribute 'KeyArray'` | JAX >= 0.4.26 removed `KeyArray` | Pin `jax==0.4.20` |
| `AttributeError: module 'scipy.linalg' has no attribute 'tril'` | SciPy >= 1.12 moved `tril` | Pin `scipy>=1.9,<1.12` |
| `AttributeError: _ARRAY_API not found` | NumPy 2.0 breaking changes | Pin `numpy>=1.22,<2.0` |
| `ModuleNotFoundError: No module named 'flax'` | Missing JAX ecosystem packages | Install flax, optax, distrax |

## Required Versions

Use these **exact versions** for guaranteed compatibility:

```
numpy>=1.22,<2.0
scipy>=1.9,<1.12
jax==0.4.20
jaxlib==0.4.20
flax==0.7.5
optax==0.1.7
distrax==0.1.5
robopianist>=1.0.6
dm_env_wrappers
wandb
tyro
tqdm
```

## Local Installation

### Option 1: Create a fresh conda environment (Recommended)

```bash
conda create -n robopianist python=3.10
conda activate robopianist

# Install JAX ecosystem first with pinned versions
pip install "numpy>=1.22,<2.0" "scipy>=1.9,<1.12"
pip install jax==0.4.20 jaxlib==0.4.20
pip install flax==0.7.5 optax==0.1.7 distrax==0.1.5

# Then install the rest
pip install robopianist>=1.0.6 dm_env_wrappers wandb tyro tqdm
```

### Option 2: Use requirements file

Create a `requirements-pinned.txt`:

```
numpy>=1.22,<2.0
scipy>=1.9,<1.12
jax==0.4.20
jaxlib==0.4.20
flax==0.7.5
optax==0.1.7
distrax==0.1.5
robopianist>=1.0.6
dm_env_wrappers
wandb
tyro
tqdm
```

Then install:

```bash
pip install -r requirements-pinned.txt
```

## Modal Deployment

In your Modal image definition, specify dependencies in this order:

```python
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "libgl1-mesa-glx",
        "libosmesa6-dev",
        "patchelf",
        "libglfw3",
        "libglew-dev",
        "ffmpeg",
        "fluidsynth",
        "git",
        "portaudio19-dev",
    )
    .env({"MUJOCO_GL": "egl"})
    .run_commands(
        "git clone https://github.com/kevinzakka/robopianist-rl /root/robopianist-rl",
    )
    .pip_install(
        # Pin numpy and scipy FIRST to prevent later packages from upgrading them
        "numpy>=1.22,<2.0",
        "scipy>=1.9,<1.12",
        # Pin JAX ecosystem versions
        "jax==0.4.20",
        "jaxlib==0.4.20",
        "flax==0.7.5",
        "optax==0.1.7",
        "distrax==0.1.5",
        # Then install the rest
        "robopianist>=1.0.6",
        "wandb",
        "tyro",
        "tqdm",
        "dm_env_wrappers",
    )
)
```

**Important:** The order matters! Install numpy/scipy constraints first, then JAX packages, then everything else. This prevents pip from upgrading to incompatible versions.

## Troubleshooting

### "Module not found" errors

Make sure all packages are installed:
```bash
pip install flax optax distrax dm_env_wrappers
```

### Version conflicts during installation

If pip complains about version conflicts, try installing in stages:
```bash
pip install "numpy>=1.22,<2.0" "scipy>=1.9,<1.12"
pip install jax==0.4.20 jaxlib==0.4.20 --no-deps
pip install flax==0.7.5 optax==0.1.7 distrax==0.1.5
pip install robopianist>=1.0.6
```

### Modal timeout errors

Modal has a maximum timeout of 86400 seconds (24 hours). If your training exceeds this, you'll need to implement checkpointing and resume functionality.

## Why These Specific Versions?

- **JAX 0.4.20**: Last version with `jax.random.KeyArray` API that robopianist-rl uses
- **NumPy < 2.0**: NumPy 2.0 introduced breaking changes to the array API
- **SciPy < 1.12**: SciPy 1.12+ moved `scipy.linalg.tril` which JAX 0.4.20 expects
- **Flax 0.7.5 / Optax 0.1.7 / Distrax 0.1.5**: Compatible with JAX 0.4.20

## Updating in the Future

If the upstream `robopianist-rl` repository updates to support newer JAX versions, you may be able to relax these constraints. Check the repo for updates periodically.
