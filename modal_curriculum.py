"""
Modal deployment for scale pretraining + NocturneRousseau fine-tuning experiment.

This runs the two-phase training:
1. Pretrain on scales (C Major, D Major - one and two hands)
2. Fine-tune on NocturneRousseau

Usage:
    modal run modal_curriculum.py                          # pretrain experiment (default)
    modal run modal_curriculum.py --experiment baseline    # baseline only
    modal run modal_curriculum.py --experiment both        # both in parallel
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
    .add_local_file("train_curriculum.py", "/root/robopianist-rl/train_curriculum.py")
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
def train_curriculum(
    pretrain_steps: int = 500_000,
    finetune_steps: int = 500_000,
    scale_switch_interval: int = 50_000,
    seed: int = 42,
    name: str = "curriculum-scales-finetune-nocturne",
):
    import os
    os.chdir("/root/robopianist-rl")

    subprocess.run([
        "python", "train_curriculum.py",
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
        "--discount", "0.8",
        "--agent-config.critic-dropout-rate", "0.01",
        "--agent-config.critic-layer-norm",
        "--agent-config.hidden-dims", "256", "256", "256",
        "--trim-silence",
        "--reduced-action-space",
        "--action-reward-observation",
        "--primitive-fingertip-collisions",
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
        "--discount", "0.8",
        "--agent-config.critic-dropout-rate", "0.01",
        "--agent-config.critic-layer-norm",
        "--agent-config.hidden-dims", "256", "256", "256",
        "--trim-silence",
        "--reduced-action-space",
        "--action-reward-observation",
        "--primitive-fingertip-collisions",
        "--root-dir", "/output",
    ], check=True)

    volume.commit()


@app.local_entrypoint()
async def main(
    experiment: str = "curriculum",
    pretrain_steps: int = 500_000,
    finetune_steps: int = 500_000,
    seed: int = 42,
):
    """
    Run experiments.

    Args:
        experiment: "curriculum" for scale pretraining experiment,
                   "baseline" for no-pretraining baseline,
                   "both" to run both in parallel
        pretrain_steps: Number of steps for pretraining phase
        finetune_steps: Number of steps for fine-tuning phase
        seed: Random seed
    """
    import asyncio
    total_steps = pretrain_steps + finetune_steps

    if experiment == "curriculum":
        await train_curriculum.remote.aio(
            pretrain_steps=pretrain_steps,
            finetune_steps=finetune_steps,
            seed=seed,
            name=f"curriculum-scales-finetune-nocturne-seed{seed}",
        )
    elif experiment == "baseline":
        await train_baseline.remote.aio(
            max_steps=total_steps,
            seed=seed,
            name=f"baseline-nocturne-no-pretrain-seed{seed}",
        )
    elif experiment == "both":
        await asyncio.gather(
            train_curriculum.remote.aio(
                pretrain_steps=pretrain_steps,
                finetune_steps=finetune_steps,
                seed=seed,
                name=f"curriculum-scales-finetune-nocturne-seed{seed}",
            ),
            train_baseline.remote.aio(
                max_steps=total_steps,
                seed=seed,
                name=f"baseline-nocturne-no-pretrain-seed{seed}",
            ),
        )
    else:
        raise ValueError(f"Unknown experiment: {experiment}. Use 'curriculum', 'baseline', or 'both'")
