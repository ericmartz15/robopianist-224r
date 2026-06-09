import modal
import subprocess
import threading
import time

app = modal.App("robopianist-eb-onset")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1-mesa-glx","libosmesa6-dev","patchelf","libglfw3","libglew-dev","ffmpeg","fluidsynth","git","portaudio19-dev")
    .env({"MUJOCO_GL": "egl", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
    .run_commands("git clone https://github.com/kevinzakka/robopianist-rl /root/robopianist-rl")
    .pip_install("numpy>=1.22,<2.0","scipy>=1.9,<1.12","jax==0.4.20","jaxlib==0.4.20","flax==0.7.5","optax==0.1.7","distrax==0.1.5","mujoco==3.7.0","dm-control==1.0.39","robopianist>=1.0.6","wandb","tyro","tqdm","dm_env_wrappers")
    .add_local_file("train_curriculum_eb.py", "/root/robopianist-rl/train_curriculum_eb.py")
    .add_local_file("onset_alignment.py", "/root/robopianist-rl/onset_alignment.py")
)

volume = modal.Volume.from_name("robopianist-results", create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb")


@app.function(image=image, volumes={"/output": volume}, gpu="A10G", timeout=86400, secrets=[wandb_secret], retries=modal.Retries(max_retries=10, initial_delay=30.0, backoff_coefficient=1.0))
def run():
    import os
    os.chdir("/root/robopianist-rl")
    proc = subprocess.Popen(["python", "train_curriculum_eb.py", "--mode", "online", "--project", "robopianist-224r", "--name", "eb-curriculum-onset-a0.1-seed42", "--pretrain_steps", "100000", "--finetune_steps", "400000", "--seed", "42", "--gravity_compensation", "--n_steps_lookahead", "10", "--tqdm_bar", "--root_dir", "/output", "--discount", "0.8", "--agent-config.critic-dropout-rate", "0.01", "--agent-config.critic-layer-norm", "--agent-config.hidden-dims", "256", "256", "256", "--trim-silence", "--reduced-action-space", "--action-reward-observation", "--primitive-fingertip-collisions", "--control_timestep", "0.05", "--pretrain_shift", "3", "--onset_alpha", "0.1", "--onset_sigma", "2.0"])
    def _commit():
        while proc.poll() is None:
            time.sleep(300)
            volume.commit()
    threading.Thread(target=_commit, daemon=True).start()
    proc.wait()
    volume.commit()


@app.local_entrypoint()
async def main():
    await run.remote.aio()
