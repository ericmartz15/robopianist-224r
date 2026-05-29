"""
Same as modal_curriculum_onset.py but with pretrain_steps reduced to 100_000
(instead of 500_000).

This is a fully self-contained Modal app: the image/volume/secret are defined
inline (not imported from modal_curriculum_onset.py), so the remote container
does not need any other local module to be present.

Usage:
    modal run --detach modal_curriculum_onset_short.py
"""

import subprocess

import modal

app = modal.App("robopianist-curriculum-onset-short")

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
    .add_local_file("train_curriculum_onset.py", "/root/robopianist-rl/train_curriculum_onset.py")
    .add_local_file("onset_alignment.py", "/root/robopianist-rl/onset_alignment.py")
)

volume = modal.Volume.from_name("robopianist-results", create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb")


@app.function(
    image=image,
    volumes={"/output": volume},
    gpu="T4",
    timeout=86400,
    secrets=[wandb_secret],
)
def train_curriculum_onset_short(
    pretrain_steps: int = 100_000,
    finetune_steps: int = 500_000,
    scale_switch_interval: int = 50_000,
    seed: int = 42,
    name: str = "curriculum-onset-short",
    onset_alpha: float = 0.1,
    onset_sigma: float = 2.0,
):
    import os
    os.chdir("/root/robopianist-rl")

    subprocess.run([
        "python", "train_curriculum_onset.py",
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
        "--onset_alpha", str(onset_alpha),
        "--onset_sigma", str(onset_sigma),
    ], check=True)

    volume.commit()


@app.local_entrypoint()
def main(
    pretrain_steps: int = 100_000,
    finetune_steps: int = 500_000,
    seed: int = 42,
    onset_alpha: float = 0.1,
    onset_sigma: float = 2.0,
):
    call = train_curriculum_onset_short.spawn(
        pretrain_steps=pretrain_steps,
        finetune_steps=finetune_steps,
        seed=seed,
        name=f"curriculum-onset-short-a{onset_alpha}-s{onset_sigma}-seed{seed}",
        onset_alpha=onset_alpha,
        onset_sigma=onset_sigma,
    )
    print(f"Job submitted. Function call ID: {call.object_id}")
    print("Monitor at https://modal.com/apps/afishpez/main/")
