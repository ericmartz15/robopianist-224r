"""
Training script with scale pretraining followed by fine-tuning on NocturneRousseau.

This implements a two-phase training approach:
1. Pretrain on scales (C Major, D Major - both one and two hands)
2. Fine-tune on NocturneRousseau

Usage:
    python train_pretrain.py --pretrain_steps 500000 --finetune_steps 500000
"""

from pathlib import Path
from typing import Optional, Tuple, List
import tyro
from dataclasses import dataclass, asdict, field
import wandb
import time
import random
import numpy as np
from tqdm import tqdm
import pickle

import sac
import specs
import replay

from robopianist import suite
import dm_env_wrappers as wrappers
import robopianist.wrappers as robopianist_wrappers


SCALE_ENVIRONMENTS = [
    "RoboPianist-debug-CMajorScaleOneHand-v0",
    "RoboPianist-debug-CMajorScaleTwoHands-v0",
    "RoboPianist-debug-DMajorScaleOneHand-v0",
    "RoboPianist-debug-DMajorScaleTwoHands-v0",
]

TARGET_ENVIRONMENT = "RoboPianist-debug-NocturneRousseau-v0"


@dataclass(frozen=True)
class Args:
    root_dir: str = "/tmp/robopianist"
    seed: int = 42
    pretrain_steps: int = 500_000
    finetune_steps: int = 500_000
    warmstart_steps: int = 5_000
    log_interval: int = 1_000
    eval_interval: int = 10_000
    eval_episodes: int = 1
    batch_size: int = 256
    discount: float = 0.99
    tqdm_bar: bool = False
    replay_capacity: int = 1_000_000
    project: str = "robopianist"
    entity: str = ""
    name: str = ""
    tags: str = ""
    notes: str = ""
    mode: str = "disabled"
    scale_environments: List[str] = field(default_factory=lambda: SCALE_ENVIRONMENTS)
    target_environment: str = TARGET_ENVIRONMENT
    n_steps_lookahead: int = 10
    trim_silence: bool = False
    gravity_compensation: bool = False
    reduced_action_space: bool = False
    control_timestep: float = 0.05
    stretch_factor: float = 1.0
    shift_factor: int = 0
    wrong_press_termination: bool = False
    disable_fingering_reward: bool = False
    disable_forearm_reward: bool = False
    disable_colorization: bool = False
    disable_hand_collisions: bool = False
    primitive_fingertip_collisions: bool = False
    frame_stack: int = 1
    clip: bool = True
    record_dir: Optional[Path] = None
    record_every: int = 1
    record_resolution: Tuple[int, int] = (480, 640)
    camera_id: Optional[str | int] = "piano/back"
    action_reward_observation: bool = False
    agent_config: sac.SACConfig = field(default_factory=sac.SACConfig)
    scale_switch_interval: int = 50_000
    clear_replay_on_finetune: bool = True
    finetune_warmstart_steps: int = 1_000


def prefix_dict(prefix: str, d: dict) -> dict:
    return {f"{prefix}/{k}": v for k, v in d.items()}


def get_env(
    environment_name: str,
    args: Args,
    seed: int,
    record_dir: Optional[Path] = None,
):
    env = suite.load(
        environment_name=environment_name,
        seed=seed,
        stretch=args.stretch_factor,
        shift=args.shift_factor,
        task_kwargs=dict(
            n_steps_lookahead=args.n_steps_lookahead,
            trim_silence=args.trim_silence,
            gravity_compensation=args.gravity_compensation,
            reduced_action_space=args.reduced_action_space,
            control_timestep=args.control_timestep,
            wrong_press_termination=args.wrong_press_termination,
            disable_fingering_reward=args.disable_fingering_reward,
            disable_forearm_reward=args.disable_forearm_reward,
            disable_colorization=args.disable_colorization,
            disable_hand_collisions=args.disable_hand_collisions,
            primitive_fingertip_collisions=args.primitive_fingertip_collisions,
            change_color_on_activation=True,
        ),
    )
    if record_dir is not None:
        env = robopianist_wrappers.PianoSoundVideoWrapper(
            environment=env,
            record_dir=record_dir,
            record_every=args.record_every,
            camera_id=args.camera_id,
            height=args.record_resolution[0],
            width=args.record_resolution[1],
        )
        env = wrappers.EpisodeStatisticsWrapper(
            environment=env, deque_size=args.record_every
        )
        env = robopianist_wrappers.MidiEvaluationWrapper(
            environment=env, deque_size=args.record_every
        )
    else:
        env = wrappers.EpisodeStatisticsWrapper(environment=env, deque_size=1)
    if args.action_reward_observation:
        env = wrappers.ObservationActionRewardWrapper(env)
    env = wrappers.ConcatObservationWrapper(env)
    if args.frame_stack > 1:
        env = wrappers.FrameStackingWrapper(
            env, num_frames=args.frame_stack, flatten=True
        )
    env = wrappers.CanonicalSpecWrapper(env, clip=args.clip)
    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.DmControlWrapper(env)
    return env


def save_checkpoint(agent: sac.SAC, path: Path) -> None:
    """Save agent checkpoint to disk."""
    checkpoint = {
        "actor_params": agent.actor.params,
        "critic_params": agent.critic.params,
        "target_critic_params": agent.target_critic.params,
        "temp_params": agent.temp.params,
        "rng": agent.rng,
    }
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(agent: sac.SAC, path: Path) -> sac.SAC:
    """Load agent checkpoint from disk."""
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)
    
    agent = agent.replace(
        actor=agent.actor.replace(params=checkpoint["actor_params"]),
        critic=agent.critic.replace(params=checkpoint["critic_params"]),
        target_critic=agent.target_critic.replace(params=checkpoint["target_critic_params"]),
        temp=agent.temp.replace(params=checkpoint["temp_params"]),
        rng=checkpoint["rng"],
    )
    print(f"Loaded checkpoint from {path}")
    return agent


def train_phase(
    agent: sac.SAC,
    env,
    eval_env,
    replay_buffer: replay.Buffer,
    spec: specs.EnvironmentSpec,
    args: Args,
    num_steps: int,
    start_step: int,
    warmstart_steps: int,
    phase_name: str,
    experiment_dir: Path,
) -> sac.SAC:
    """Run a training phase."""
    timestep = env.reset()
    replay_buffer.insert(timestep, None)
    
    start_time = time.time()
    
    for i in tqdm(range(1, num_steps + 1), disable=not args.tqdm_bar, desc=phase_name):
        global_step = start_step + i
        
        if i < warmstart_steps:
            action = spec.sample_action(random_state=env.random_state)
        else:
            agent, action = agent.sample_actions(timestep.observation)
        
        timestep = env.step(action)
        replay_buffer.insert(timestep, action)
        
        if timestep.last():
            wandb.log(prefix_dict(f"{phase_name}/train", env.get_statistics()), step=global_step)
            timestep = env.reset()
            replay_buffer.insert(timestep, None)
        
        if i >= warmstart_steps:
            if replay_buffer.is_ready():
                transitions = replay_buffer.sample()
                agent, metrics = agent.update(transitions)
                if i % args.log_interval == 0:
                    wandb.log(prefix_dict(f"{phase_name}/train", metrics), step=global_step)
        
        if i % args.eval_interval == 0:
            for _ in range(args.eval_episodes):
                eval_timestep = eval_env.reset()
                while not eval_timestep.last():
                    eval_timestep = eval_env.step(agent.eval_actions(eval_timestep.observation))
            log_dict = prefix_dict(f"{phase_name}/eval", eval_env.get_statistics())
            music_dict = prefix_dict(f"{phase_name}/eval", eval_env.get_musical_metrics())
            wandb.log(log_dict | music_dict, step=global_step)
            video = wandb.Video(str(eval_env.latest_filename), fps=4, format="mp4")
            wandb.log({f"{phase_name}/video": video, "global_step": global_step})
            eval_env.latest_filename.unlink()
        
        if i % args.log_interval == 0:
            wandb.log({f"{phase_name}/train/fps": int(i / (time.time() - start_time))}, step=global_step)
    
    return agent


def pretrain_on_scales(
    agent: sac.SAC,
    args: Args,
    spec: specs.EnvironmentSpec,
    replay_buffer: replay.Buffer,
    experiment_dir: Path,
) -> sac.SAC:
    """Pretrain agent on scale environments, cycling through them."""
    print("\n" + "=" * 60)
    print("PHASE 1: PRETRAINING ON SCALES")
    print("=" * 60)
    
    num_scales = len(args.scale_environments)
    steps_per_scale = args.scale_switch_interval
    total_cycles = args.pretrain_steps // (num_scales * steps_per_scale)
    remaining_steps = args.pretrain_steps % (num_scales * steps_per_scale)
    
    current_step = 0
    
    for cycle in range(total_cycles + 1):
        for scale_idx, scale_env_name in enumerate(args.scale_environments):
            if current_step >= args.pretrain_steps:
                break
            
            steps_this_round = min(
                steps_per_scale,
                args.pretrain_steps - current_step
            )
            
            if steps_this_round <= 0:
                break
            
            print(f"\nCycle {cycle + 1}, Scale: {scale_env_name}")
            print(f"Training for {steps_this_round} steps (total: {current_step} -> {current_step + steps_this_round})")
            
            env = get_env(scale_env_name, args, args.seed + scale_idx)
            eval_env = get_env(
                scale_env_name, 
                args, 
                args.seed + scale_idx + 1000,
                record_dir=experiment_dir / "pretrain_eval"
            )
            
            warmstart = args.warmstart_steps if current_step == 0 else 0
            
            agent = train_phase(
                agent=agent,
                env=env,
                eval_env=eval_env,
                replay_buffer=replay_buffer,
                spec=spec,
                args=args,
                num_steps=steps_this_round,
                start_step=current_step,
                warmstart_steps=warmstart,
                phase_name=f"pretrain/{scale_env_name.split('-')[-2]}",
                experiment_dir=experiment_dir,
            )
            
            current_step += steps_this_round
    
    checkpoint_path = experiment_dir / "pretrain_checkpoint.pkl"
    save_checkpoint(agent, checkpoint_path)
    
    return agent


def finetune_on_target(
    agent: sac.SAC,
    args: Args,
    spec: specs.EnvironmentSpec,
    replay_buffer: replay.Buffer,
    experiment_dir: Path,
) -> sac.SAC:
    """Fine-tune the pretrained agent on the target environment."""
    print("\n" + "=" * 60)
    print("PHASE 2: FINE-TUNING ON TARGET (NocturneRousseau)")
    print("=" * 60)
    
    if args.clear_replay_on_finetune:
        print("Clearing replay buffer for fine-tuning phase...")
        replay_buffer = replay.Buffer(
            state_dim=spec.observation_dim,
            action_dim=spec.action_dim,
            max_size=args.replay_capacity,
            batch_size=args.batch_size,
        )
    
    env = get_env(args.target_environment, args, args.seed + 500)
    eval_env = get_env(
        args.target_environment,
        args,
        args.seed + 501,
        record_dir=experiment_dir / "finetune_eval"
    )
    
    agent = train_phase(
        agent=agent,
        env=env,
        eval_env=eval_env,
        replay_buffer=replay_buffer,
        spec=spec,
        args=args,
        num_steps=args.finetune_steps,
        start_step=args.pretrain_steps,
        warmstart_steps=args.finetune_warmstart_steps,
        phase_name="finetune",
        experiment_dir=experiment_dir,
    )
    
    checkpoint_path = experiment_dir / "finetune_checkpoint.pkl"
    save_checkpoint(agent, checkpoint_path)
    
    return agent


def main(args: Args) -> None:
    if args.name:
        run_name = args.name
    else:
        run_name = f"SAC-pretrain-finetune-{args.seed}-{time.time()}"
    
    experiment_dir = Path(args.root_dir) / run_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "pretrain_eval").mkdir(exist_ok=True)
    (experiment_dir / "finetune_eval").mkdir(exist_ok=True)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    config = asdict(args)
    config["scale_environments"] = list(args.scale_environments)
    
    wandb.init(
        project=args.project,
        entity=args.entity or None,
        tags=(args.tags.split(",") if args.tags else []) + ["pretrain-finetune"],
        notes=args.notes or None,
        config=config,
        mode=args.mode,
        name=run_name,
    )
    
    print("Initializing agent with target environment spec...")
    init_env = get_env(args.target_environment, args, args.seed)
    spec = specs.EnvironmentSpec.make(init_env)
    
    agent = sac.SAC.initialize(
        spec=spec,
        config=args.agent_config,
        seed=args.seed,
        discount=args.discount,
    )
    
    replay_buffer = replay.Buffer(
        state_dim=spec.observation_dim,
        action_dim=spec.action_dim,
        max_size=args.replay_capacity,
        batch_size=args.batch_size,
    )
    
    agent = pretrain_on_scales(
        agent=agent,
        args=args,
        spec=spec,
        replay_buffer=replay_buffer,
        experiment_dir=experiment_dir,
    )
    
    agent = finetune_on_target(
        agent=agent,
        args=args,
        spec=spec,
        replay_buffer=replay_buffer,
        experiment_dir=experiment_dir,
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Checkpoints saved to: {experiment_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))
