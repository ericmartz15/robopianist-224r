"""
Same as modal_curriculum_onset.py but with pretrain_steps reduced by 1/3
(333_333 instead of 500_000).

Usage:
    modal run --detach modal_curriculum_onset_short.py
"""

from modal_curriculum_onset import image, volume, wandb_secret, train_curriculum_onset
import modal

app = modal.App("robopianist-curriculum-onset-short")


@app.local_entrypoint()
def main(
    pretrain_steps: int = 100_000,
    finetune_steps: int = 500_000,
    seed: int = 42,
    onset_alpha: float = 0.1,
    onset_sigma: float = 2.0,
):
    call = train_curriculum_onset.spawn(
        pretrain_steps=pretrain_steps,
        finetune_steps=finetune_steps,
        seed=seed,
        name=f"curriculum-onset-short-a{onset_alpha}-s{onset_sigma}-seed{seed}",
        onset_alpha=onset_alpha,
        onset_sigma=onset_sigma,
    )
    print(f"Job submitted. Function call ID: {call.object_id}")
    print("Monitor at https://modal.com/apps/j-oliver-choo/main/")
