import modal
import subprocess
import threading
import time

app = modal.App("robopianist-arpeggio-curriculum")

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
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
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
    .add_local_file("arpeggio_midi.py", "/root/robopianist-rl/arpeggio_midi.py")
)

volume = modal.Volume.from_name("robopianist-results", create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb")

_fn_kwargs = dict(
    image=image,
    volumes={"/output": volume},
    gpu="A10G",
    timeout=86400,
    secrets=[wandb_secret],
    retries=modal.Retries(max_retries=10, initial_delay=30.0, backoff_coefficient=1.0),
)

ARPEGGIO_SCALE_ENVS = [
    "CMajorArpeggioOneHand",
    "CMajorArpeggioTwoHands",
    "DMajorArpeggioOneHand",
    "DMajorArpeggioTwoHands",
]


def _run(cmd: list, vol: modal.Volume) -> None:
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


def _curriculum_cmd(
    name: str,
    pretrain_steps: int,
    finetune_steps: int,
    seed: int,
    onset_alpha: float,
    onset_sigma: float,
    pretrain_shift: int,
) -> list:
    return [
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
        "--scale_environments", *ARPEGGIO_SCALE_ENVS,
        "--pretrain_shift", str(pretrain_shift),
    ]


# C/D arpeggio but no onset 

@app.function(**_fn_kwargs)
def run_arpeggio_cd_no_onset(
    pretrain_steps: int = 100_000,
    finetune_steps: int = 400_000,
    seed: int = 42,
    name: str = "",
):
    import os
    os.chdir("/root/robopianist-rl")
    _run(_curriculum_cmd(
        name=name,
        pretrain_steps=pretrain_steps,
        finetune_steps=finetune_steps,
        seed=seed,
        onset_alpha=0.0,
        onset_sigma=2.0,
        pretrain_shift=0,
    ), volume)


# C/D arpeggio and onset

@app.function(**_fn_kwargs)
def run_arpeggio_cd_onset(
    pretrain_steps: int = 100_000,
    finetune_steps: int = 400_000,
    seed: int = 42,
    onset_alpha: float = 0.1,
    onset_sigma: float = 2.0,
    name: str = "",
):
    import os
    os.chdir("/root/robopianist-rl")
    _run(_curriculum_cmd(
        name=name,
        pretrain_steps=pretrain_steps,
        finetune_steps=finetune_steps,
        seed=seed,
        onset_alpha=onset_alpha,
        onset_sigma=onset_sigma,
        pretrain_shift=0,
    ), volume)


# Eb/F arpeggio but no onset

@app.function(**_fn_kwargs)
def run_arpeggio_eb_no_onset(
    pretrain_steps: int = 100_000,
    finetune_steps: int = 400_000,
    seed: int = 42,
    name: str = "",
):
    import os
    os.chdir("/root/robopianist-rl")
    _run(_curriculum_cmd(
        name=name,
        pretrain_steps=pretrain_steps,
        finetune_steps=finetune_steps,
        seed=seed,
        onset_alpha=0.0,
        onset_sigma=2.0,
        pretrain_shift=3,  # C/D to Eb/F
    ), volume)


# Eb/F arpeggio and onset 

@app.function(**_fn_kwargs)
def run_arpeggio_eb_onset(
    pretrain_steps: int = 100_000,
    finetune_steps: int = 400_000,
    seed: int = 42,
    onset_alpha: float = 0.1,
    onset_sigma: float = 2.0,
    name: str = "",
):
    import os
    os.chdir("/root/robopianist-rl")
    _run(_curriculum_cmd(
        name=name,
        pretrain_steps=pretrain_steps,
        finetune_steps=finetune_steps,
        seed=seed,
        onset_alpha=onset_alpha,
        onset_sigma=onset_sigma,
        pretrain_shift=3,
    ), volume)

@app.local_entrypoint()
def main(seed: int = 42, onset_alpha: float = 0.1, onset_sigma: float = 2.0):
    
    fc1 = run_arpeggio_cd_no_onset.spawn(
        seed=seed, name=f"arpeggio-cd-no-onset-seed{seed}")
    fc2 = run_arpeggio_cd_onset.spawn(
        seed=seed, onset_alpha=onset_alpha, onset_sigma=onset_sigma,
        name=f"arpeggio-cd-onset-a{onset_alpha}-seed{seed}")
    fc3 = run_arpeggio_eb_no_onset.spawn(
        seed=seed, name=f"arpeggio-eb-no-onset-seed{seed}")
    fc4 = run_arpeggio_eb_onset.spawn(
        seed=seed, onset_alpha=onset_alpha, onset_sigma=onset_sigma,
        name=f"arpeggio-eb-onset-a{onset_alpha}-seed{seed}")
