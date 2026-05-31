"""
Same 2x2 curriculum experiments as modal_experiments.py but with
100k pretraining + 500k finetuning (600k total) instead of 100k + 400k.

Usage:
    modal run --detach modal_more_finetune.py
"""

from modal_experiments import app, run_curriculum, run_curriculum_onset


@app.local_entrypoint()
def main(seed: int = 42, onset_alpha: float = 0.1, onset_sigma: float = 2.0):
    fc1 = run_curriculum.spawn(
        pretrain_steps=100_000,
        finetune_steps=500_000,
        seed=seed,
        name=f"curriculum-no-onset-600k-seed{seed}",
    )
    fc2 = run_curriculum_onset.spawn(
        pretrain_steps=100_000,
        finetune_steps=500_000,
        seed=seed,
        onset_alpha=onset_alpha,
        onset_sigma=onset_sigma,
        name=f"curriculum-onset-600k-a{onset_alpha}-seed{seed}",
    )
    print(f"Spawned 2 experiments with 100k pretrain + 500k finetune (seed={seed}).")
    print(f"  curriculum (no onset): {fc1.object_id}")
    print(f"  curriculum + onset:    {fc2.object_id}")
