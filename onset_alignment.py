"""Reward wrapper for temporal onset alignment.

Augments the base F1 reward with a per-onset Gaussian bonus:

    bonus = mean over detected onsets of exp(-dt² / 2σ²)

where dt is the distance in timesteps to the nearest expected onset for that
key in the MIDI score. The bonus is in [0, 1], weighted by `alpha`.
"""

import numpy as np
import dm_env
import dm_env_wrappers as wrappers


class OnsetAlignmentWrapper(wrappers.EnvironmentWrapper):
    """Adds a temporal onset-alignment bonus to the environment reward.

    Args:
        environment:       The wrapped dm_env environment (directly from suite.load()).
        midi_seq:          A NoteSequence (i.e. midi_file.seq) with note onset times.
        control_timestep:  Duration of each simulation step in seconds.
        alpha:             Weight of the onset bonus (base reward is unscaled).
        sigma:             Temporal tolerance in timesteps; 1σ ≈ sigma * control_timestep s.
    """

    def __init__(
        self,
        environment: dm_env.Environment,
        midi_seq,
        control_timestep: float,
        alpha: float = 0.1,
        sigma: float = 2.0,
    ):
        super().__init__(environment)
        self._alpha = alpha
        self._sigma = sigma
        self._step = 0
        self._prev_keys = None
        self._onset_steps = self._build_onset_steps(midi_seq, control_timestep)
        self.last_bonus = 0.0  # exposed for per-step logging

    @staticmethod
    def _build_onset_steps(seq, dt: float) -> list:
        """For each of 88 piano keys, a sorted int array of expected onset steps."""
        buckets = [[] for _ in range(88)]
        for note in seq.notes:
            buckets[note.pitch - 21].append(round(note.start_time / dt))
        return [np.array(b, dtype=np.int32) for b in buckets]

    def reset(self) -> dm_env.TimeStep:
        self._step = 0
        self._prev_keys = None
        return self._environment.reset()

    def step(self, action) -> dm_env.TimeStep:
        ts = self._environment.step(action)
        # Read key activations directly from the task's piano entity —
        # avoids any dependency on observation key names.
        keys = self._environment.task.piano.activation.astype(np.int8)

        bonus = 0.0
        if self._prev_keys is not None:
            onset_indices = np.where((keys == 1) & (self._prev_keys == 0))[0]
            if len(onset_indices):
                gaussians = []
                for k in onset_indices:
                    targets = self._onset_steps[k]
                    if len(targets):
                        dt = float(np.min(np.abs(targets - self._step)))
                        gaussians.append(np.exp(-0.5 * (dt / self._sigma) ** 2))
                if gaussians:
                    bonus = float(np.mean(gaussians))  # in [0, 1]

        self.last_bonus = bonus
        self._prev_keys = keys
        self._step += 1
        return ts._replace(reward=ts.reward + self._alpha * bonus)
