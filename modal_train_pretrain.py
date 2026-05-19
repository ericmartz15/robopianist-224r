"""
Modal deployment for scale pretraining + NocturneRousseau fine-tuning experiment.

This runs the two-phase training:
1. Pretrain on scales (C Major, D Major - one and two hands)
2. Fine-tune on NocturneRousseau

Usage:
    modal run modal_train_pretrain.py
"""

import modal
import subprocess

app = modal.App("robopianist-pretrain-finetune")

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
        "numpy>=1.22,<2.0",
        "scipy>=1.9,<1.12",
        "jax==0.4.20",
        "jaxlib==0.4.20",
        "flax==0.7.5",
        "optax==0.1.7",
        "distrax==0.1.5",
        "robopianist>=1.0.6",
        "wandb",
        "tyro",
        "tqdm",
        "dm_env_wrappers",
    )
    .add_local_file("train_pretrain.py", "/root/robopianist-rl/train_pretrain.py")
)

volume = modal.Volume.from_name("robopianist-results", create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb")


@app.function(
    image=image,
    volumes={"/output": volume},
    gpu="T4",
    timeout=86400,  # 24 hours (Modal maximum)
    secrets=[wandb_secret],
)
def train_pretrain_finetune(
    pretrain_steps: int = 500_000,
    finetune_steps: int = 500_000,
    scale_switch_interval: int = 50_000,
    seed: int = 42,
    name: str = "pretrain-scales-finetune-nocturne",
):
    import os
    os.chdir("/root/robopianist-rl")
    
    subprocess.run([
        "python", "train_pretrain.py",
        "--mode", "online",
        "--project", "robopianist-224r",
        "--name", name,
        "--pretrain_steps", str(pretrain_steps),
        "--finetune_steps", str(finetune_steps),
        "--scale_switch_interval", str(scale_switch_interval),
        "--seed", str(seed),
        "--gravity_compensation",
        "--n_steps_lookahead", "10",
        "--tqdm_bar",
        "--root_dir", "/output",
    ], check=True)
    
    volume.commit()


@app.function(
    image=image,
    volumes={"/output": volume},
    gpu="T4",
    timeout=86400,  # 24 hours (Modal maximum)
    secrets=[wandb_secret],
)
def train_baseline(
    max_steps: int = 1_000_000,
    seed: int = 42,
    name: str = "baseline-nocturne-no-pretrain",
):
    """Train baseline without pretraining for comparison."""
    import os
    os.chdir("/root/robopianist-rl")
    
    subprocess.run([
        "python", "train.py",
        "--mode", "online",
        "--project", "robopianist-224r",
        "--name", name,
        "--max_steps", str(max_steps),
        "--environment_name", "RoboPianist-debug-NocturneRousseau-v0",
        "--gravity_compensation",
        "--n_steps_lookahead", "10",
        "--tqdm_bar",
    ], check=True)
    
    volume.commit()


@app.local_entrypoint()
def main(
    experiment: str = "pretrain",
    pretrain_steps: int = 500_000,
    finetune_steps: int = 500_000,
    seed: int = 42,
):
    """
    Run experiments.
    
    Args:
        experiment: "pretrain" for scale pretraining experiment, 
                   "baseline" for no-pretraining baseline,
                   "both" to run both in parallel
        pretrain_steps: Number of steps for pretraining phase
        finetune_steps: Number of steps for fine-tuning phase
        seed: Random seed
    """
    if experiment == "pretrain":
        train_pretrain_finetune.remote(
            pretrain_steps=pretrain_steps,
            finetune_steps=finetune_steps,
            seed=seed,
            name=f"pretrain-scales-finetune-nocturne-seed{seed}",
        )
    elif experiment == "baseline":
        total_steps = pretrain_steps + finetune_steps
        train_baseline.remote(
            max_steps=total_steps,
            seed=seed,
            name=f"baseline-nocturne-no-pretrain-seed{seed}",
        )
    elif experiment == "both":
        total_steps = pretrain_steps + finetune_steps
        train_pretrain_finetune.remote(
            pretrain_steps=pretrain_steps,
            finetune_steps=finetune_steps,
            seed=seed,
            name=f"pretrain-scales-finetune-nocturne-seed{seed}",
        )
        train_baseline.remote(
            max_steps=total_steps,
            seed=seed,
            name=f"baseline-nocturne-no-pretrain-seed{seed}",
        )
    else:
        raise ValueError(f"Unknown experiment: {experiment}. Use 'pretrain', 'baseline', or 'both'")
