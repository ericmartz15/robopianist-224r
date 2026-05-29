"""
Unified experiment runner — all four 500k-compute experiments in parallel.

2x2 design (identical hyperparameters, same total compute budget):
                     no onset reward    onset reward
  no curriculum      baseline           onset-only
  curriculum         curriculum         curriculum+onset

Usage:
    modal run modal_experiments.py              # all four, seed=42
    modal run modal_experiments.py --seed 1     # different seed
"""

import modal
import subprocess
import threading
import time

app = modal.App("robopianist-experiments")

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
    .env({
        "MUJOCO_GL": "egl",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",  # match run.sh; prevents JAX pre-allocating all GPU memory
    })
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
        # Pin mujoco + dm-control together: dm-control>=1.0.40 references
        # flex_bandwidth which was removed in mujoco 3.9.0, causing an
        # AttributeError at environment init. 1.0.39 is the last safe version;
        # it requires mujoco>=3.7.0 and was designed for that combination.
        "mujoco==3.7.0",
        "dm-control==1.0.39",
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

# Shared decorator kwargs — all four experiments use identical infrastructure.
_fn_kwargs = dict(
    image=image,
    volumes={"/output": volume},
    gpu="A10G",
    timeout=86400,
    secrets=[wandb_secret],
    retries=modal.Retries(max_retries=10, initial_delay=30.0, backoff_coefficient=1.0),
)


def _run(cmd: list, vol: modal.Volume) -> None:
    """Run a training subprocess and commit the volume every 5 min so
    checkpoints survive GPU preemption between explicit retries."""
    proc = subprocess.Popen(cmd)

    def _commit_loop():
        while proc.poll() is None:
            time.sleep(300)
            vol.commit()

    threading.Thread(target=_commit_loop, daemon=True).start()
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd[1])
    vol.commit()


# ── Experiment 1: Baseline ────────────────────────────────────────────────────

@app.function(**_fn_kwargs)
def run_baseline(max_steps: int = 500_000, seed: int = 42, name: str = ""):
    """No curriculum, no onset reward. Uses train_curriculum_onset.py with
    pretrain_steps=0 and onset_alpha=0 so all four conditions log identical
    metrics (onset_bonus_per_step is computed but not added to reward)."""
    import os
    os.chdir("/root/robopianist-rl")
    _run([
        "python", "train_curriculum_onset.py",
        "--mode", "online",
        "--project", "robopianist-224r",
        "--name", name,
        "--pretrain_steps", "0",
        "--finetune_steps", str(max_steps),
        "--finetune_warmstart_steps", "5000",
        "--seed", str(seed),
        "--gravity_compensation",
        "--n_steps_lookahead", "10",
        "--control_timestep", "0.05",
        "--tqdm_bar",
        "--discount", "0.8",
        "--agent-config.critic-dropout-rate", "0.01",
        "--agent-config.critic-layer-norm",
        "--agent-config.hidden-dims", "256", "256", "256",
        "--trim-silence",
        "--reduced-action-space",
        "--action-reward-observation",
        "--primitive-fingertip-collisions",
        "--root_dir", "/output",
        "--onset_alpha", "0.0",   # wrapper runs but adds 0 reward; bonus still logged
    ], volume)


# ── Experiment 2: Curriculum (no onset reward) ────────────────────────────────

@app.function(**_fn_kwargs)
def run_curriculum(
    pretrain_steps: int = 100_000,
    finetune_steps: int = 400_000,
    seed: int = 42,
    name: str = "",
):
    import os
    os.chdir("/root/robopianist-rl")
    _run([
        "python", "train_curriculum_onset.py",
        "--mode", "online",
        "--project", "robopianist-224r",
        "--name", name,
        "--pretrain_steps", str(pretrain_steps),
        "--finetune_steps", str(finetune_steps),
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
        "--control_timestep", "0.05",
        "--onset_alpha", "0.0",   # onset_alpha=0 → no onset bonus, pure curriculum
    ], volume)


# ── Experiment 3: Onset reward only (no curriculum) ──────────────────────────

@app.function(**_fn_kwargs)
def run_onset_only(
    finetune_steps: int = 500_000,
    seed: int = 42,
    onset_alpha: float = 0.1,
    onset_sigma: float = 2.0,
    name: str = "",
):
    """500k steps on NocturneRousseau with onset reward, no scale pretraining."""
    import os
    os.chdir("/root/robopianist-rl")
    _run([
        "python", "train_curriculum_onset.py",
        "--mode", "online",
        "--project", "robopianist-224r",
        "--name", name,
        "--pretrain_steps", "0",        # skip pretrain phase entirely
        "--finetune_steps", str(finetune_steps),
        "--finetune_warmstart_steps", "5000",  # match baseline warmstart
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
        "--control_timestep", "0.05",
        "--onset_alpha", str(onset_alpha),
        "--onset_sigma", str(onset_sigma),
    ], volume)


# ── Experiment 4: Curriculum + temporal onset reward ─────────────────────────

@app.function(**_fn_kwargs)
def run_curriculum_onset(
    pretrain_steps: int = 100_000,
    finetune_steps: int = 400_000,
    seed: int = 42,
    onset_alpha: float = 0.1,
    onset_sigma: float = 2.0,
    name: str = "",
):
    import os
    os.chdir("/root/robopianist-rl")
    _run([
        "python", "train_curriculum_onset.py",
        "--mode", "online",
        "--project", "robopianist-224r",
        "--name", name,
        "--pretrain_steps", str(pretrain_steps),
        "--finetune_steps", str(finetune_steps),
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
        "--control_timestep", "0.05",
        "--onset_alpha", str(onset_alpha),
        "--onset_sigma", str(onset_sigma),
    ], volume)


# ── Entrypoint ────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(seed: int = 42, onset_alpha: float = 0.1, onset_sigma: float = 2.0):
    """
    Spawn all four 500k-compute experiments in parallel. Returns immediately.
    Each run auto-retries on GPU preemption, resuming from its last checkpoint.

    2x2 design:
                       no onset reward    onset reward
      no curriculum    baseline           onset-only
      curriculum       curriculum         curriculum+onset

    Monitor progress at https://wandb.ai/
    """
    fc1 = run_baseline.spawn(
        max_steps=500_000,
        seed=seed,
        name=f"baseline-500k-seed{seed}",
    )
    fc2 = run_onset_only.spawn(
        finetune_steps=500_000,
        seed=seed,
        onset_alpha=onset_alpha,
        onset_sigma=onset_sigma,
        name=f"onset-only-a{onset_alpha}-seed{seed}",
    )
    fc3 = run_curriculum.spawn(
        pretrain_steps=100_000,
        finetune_steps=400_000,
        seed=seed,
        name=f"curriculum-no-onset-seed{seed}",
    )
    fc4 = run_curriculum_onset.spawn(
        pretrain_steps=100_000,
        finetune_steps=400_000,
        seed=seed,
        onset_alpha=onset_alpha,
        onset_sigma=onset_sigma,
        name=f"curriculum-onset-a{onset_alpha}-seed{seed}",
    )
    print(f"Spawned 4 experiments in parallel (seed={seed}, onset_alpha={onset_alpha}).")
    print(f"  baseline:              {fc1.object_id}")
    print(f"  onset only:            {fc2.object_id}")
    print(f"  curriculum (no onset): {fc3.object_id}")
    print(f"  curriculum + onset:    {fc4.object_id}")
    print("All runs will auto-retry on preemption and resume from checkpoints.")
