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
        "robopianist>=1.0.6",
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
        "--max_steps", "1000000",
        "--gravity_compensation", "True",
        "--n_steps_lookahead", "10",
        "--tqdm_bar", "True",
    ], check=True)
    volume.commit()

@app.local_entrypoint()
def main():
    train.remote()