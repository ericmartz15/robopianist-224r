import modal
import subprocess
import threading
import time

app = modal.App("robopianist-alpha-sweep")

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


@app.function(**_fn_kwargs)
def run_onset_sweep(onset_alpha: float, seed: int = 42):
    import os
    os.chdir("/root/robopianist-rl")
    _run([
        "python", "train_curriculum_onset.py",
        "--mode", "online",
        "--project", "robopianist-224r",
        "--name", f"onset-only-a{onset_alpha}-seed{seed}",
        "--pretrain_steps", "0",
        "--finetune_steps", "500000",
        "--finetune_warmstart_steps", "5000",
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
        "--onset_sigma", "2.0",
    ], volume)


@app.local_entrypoint()
def main(seed: int = 42, skip_existing: bool = True):
    alphas = [0.01, 0.05, 0.1, 0.25, 0.5]

    if skip_existing:
        alphas = [a for a in alphas if a != 0.1]

    handles = {}
    for alpha in alphas:
        fc = run_onset_sweep.spawn(onset_alpha=alpha, seed=seed)
        handles[alpha] = fc.object_id

    for alpha, fid in handles.items():
        print(f"  onset_alpha={alpha:<5}  {fid}")
    
