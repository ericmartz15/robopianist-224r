"""
Curriculum experiment with Eb major scale pretraining (shift=3 semitones).

The Nocturne is in Eb major, so this tests whether key-matched pretraining
helps more than the mismatched C/D major scales used in modal_experiments.py.
Runs both curriculum conditions (with and without onset reward) in parallel.

Usage:
    modal run --detach modal_eb_curriculum.py
"""

import modal
from modal_experiments import image as _base_image, volume, wandb_secret, _fn_kwargs, _run

app = modal.App("robopianist-eb-curriculum")

# Extend the existing image with the Eb-aware training script.
image = _base_image.add_local_file(
    "train_curriculum_eb.py", "/root/robopianist-rl/train_curriculum_eb.py"
)
_eb_fn_kwargs = {**_fn_kwargs, "image": image}


@app.function(**_eb_fn_kwargs)
def run_eb_curriculum(
    pretrain_steps: int = 100_000,
    finetune_steps: int = 400_000,
    seed: int = 42,
    name: str = "",
):
    """Eb major scale pretraining, no onset reward."""
    import os
    os.chdir("/root/robopianist-rl")
    _run([
        "python", "train_curriculum_eb.py",
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
        "--pretrain_shift", "3",
        "--onset_alpha", "0.0",
    ], volume)


@app.function(**_eb_fn_kwargs)
def run_eb_curriculum_onset(
    pretrain_steps: int = 100_000,
    finetune_steps: int = 400_000,
    seed: int = 42,
    onset_alpha: float = 0.1,
    onset_sigma: float = 2.0,
    name: str = "",
):
    """Eb major scale pretraining with onset-alignment reward."""
    import os
    os.chdir("/root/robopianist-rl")
    _run([
        "python", "train_curriculum_eb.py",
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
        "--pretrain_shift", "3",
        "--onset_alpha", str(onset_alpha),
        "--onset_sigma", str(onset_sigma),
    ], volume)


@app.local_entrypoint()
def main(
    pretrain_steps: int = 100_000,
    finetune_steps: int = 400_000,
    seed: int = 42,
    onset_alpha: float = 0.1,
    onset_sigma: float = 2.0,
):
    fc1 = run_eb_curriculum.spawn(
        pretrain_steps=pretrain_steps,
        finetune_steps=finetune_steps,
        seed=seed,
        name=f"eb-curriculum-no-onset-seed{seed}",
    )
    fc2 = run_eb_curriculum_onset.spawn(
        pretrain_steps=pretrain_steps,
        finetune_steps=finetune_steps,
        seed=seed,
        onset_alpha=onset_alpha,
        onset_sigma=onset_sigma,
        name=f"eb-curriculum-onset-a{onset_alpha}-seed{seed}",
    )
    print(f"Spawned 2 Eb-major curriculum experiments (seed={seed}).")
    print(f"  curriculum (no onset): {fc1.object_id}")
    print(f"  curriculum + onset:    {fc2.object_id}")
