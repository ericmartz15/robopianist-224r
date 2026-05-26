import modal
import subprocess

app = modal.App("robopianist-224r")

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
        "dm_env_wrappers",
        "wandb",
        "tyro",
        "tqdm",
    )
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
def train():
    import os
    os.chdir("/root/robopianist-rl")
    subprocess.run([
        "python", "train.py",
        "--mode", "online",
        "--project", "robopianist-224r",
        "--name", "baseline-twinkle",
        "--max_steps", "5000000",
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

@app.function(
    image=image,
    volumes={"/output": volume},
    gpu="T4",
    timeout=600,
    secrets=[wandb_secret],
)
def smoke_test():
    import os
    os.chdir("/root/robopianist-rl")
    subprocess.run([
        "python", "train.py",
        "--warmstart-steps", "5000",
        "--max-steps", "6000",
        "--discount", "0.8",
        "--agent-config.critic-dropout-rate", "0.01",
        "--agent-config.critic-layer-norm",
        "--agent-config.hidden-dims", "256", "256", "256",
        "--trim-silence",
        "--gravity-compensation",
        "--reduced-action-space",
        "--control-timestep", "0.05",
        "--n-steps-lookahead", "10",
        "--environment-name", "RoboPianist-debug-TwinkleTwinkleRousseau-v0",
        "--action-reward-observation",
        "--primitive-fingertip-collisions",
        "--tqdm-bar",
        "--root-dir", "/output",
    ], check=True)

@app.local_entrypoint()
def main(run: str = "train"):
    if run == "smoke":
        smoke_test.remote()
    else:
        train.remote()