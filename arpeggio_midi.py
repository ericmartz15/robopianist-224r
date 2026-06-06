"""
Arpeggio MIDI generators and direct environment loader for RoboPianist.

IMPORTANT: arpeggio environments must NOT be loaded via suite.load(midi_file=path)
because saving a NoteSequence to .mid and reloading loses the `part` (fingering)
field — MIDI channels don't map back to robopianist part numbers. This causes
has_fingering() to return False, disabling the 10-dim fingering observable and
producing a 1154-dim observation instead of the expected 1164-dim one.

Instead, use make_arpeggio_env() which instantiates PianoWithShadowHands directly
from the in-memory MidiFile, preserving part values and matching the target
environment's observation space exactly.

Broken-chord (arpeggio) patterns in C and D major. These are structurally
closer to the Nocturne's left-hand accompaniment (waltz-bass arpeggios) than
scale exercises, while still being simpler than the target piece.

Use shift=3 via suite.load to get Eb/F major versions (key-matched to
NocturneRousseau).

Environments:
    CMajorArpeggioOneHand    - RH broken chord, C-E-G-C up/down
    CMajorArpeggioTwoHands   - waltz-bass: LH root, RH chord tones
    DMajorArpeggioOneHand    - RH broken chord, D-F#-A-D up/down
    DMajorArpeggioTwoHands   - waltz-bass in D major
"""

from pathlib import Path
from typing import Callable, Dict, Optional

import dm_env
from note_seq import music_pb2
from robopianist.music import midi_file
from robopianist.suite.tasks import piano_with_shadow_hands
from mujoco_utils import composer_utils


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def _c_major_arpeggio_one_hand(
    right_octave: int = 6,
    note_duration: float = 0.25,
) -> midi_file.MidiFile:
    """C major broken chord, right hand. C-E-G-C up and down, 4 repetitions."""
    seq = music_pb2.NoteSequence()
    seq.sequence_metadata.title = "C major arpeggio (one hand)"
    seq.sequence_metadata.artist = "robopianist"

    root = 12 * right_octave  # C6 = 72
    # C E G C(up) G E C(down)
    pattern = [0, 4, 7, 12, 7, 4, 0]
    fingering = [0, 1, 2, 4, 2, 1, 0]

    notes = pattern * 4
    fingers = fingering * 4

    for i, (pitch, finger) in enumerate(zip(notes, fingers)):
        seq.notes.add(
            pitch=root + pitch,
            start_time=i * note_duration,
            end_time=(i + 1) * note_duration,
            velocity=80,
            part=finger,
        )

    seq.total_time = len(notes) * note_duration
    seq.tempos.add(qpm=120)
    return midi_file.MidiFile(seq=seq)


def _c_major_arpeggio_two_hands(
    left_octave: int = 4,
    right_octave: int = 6,
    note_duration: float = 0.25,
) -> midi_file.MidiFile:
    """
    C major waltz-bass arpeggio (two hands), 6 cycles.
    LH plays the bass note on beat 1; RH plays E-G-C on beats 2-4.
    Mimics the Nocturne's left-hand accompaniment pattern.
    """
    seq = music_pb2.NoteSequence()
    seq.sequence_metadata.title = "C major arpeggio (two hands)"
    seq.sequence_metadata.artist = "robopianist"

    lh_bass = 12 * left_octave        # C4 = 48
    rh_root = 12 * right_octave       # C6 = 72
    rh_pattern = [4, 7, 12]           # E G C (above root)
    rh_fingers = [1, 2, 4]
    lh_finger = 9                     # LH thumb

    for cycle in range(6):
        t = cycle * 4 * note_duration
        # LH bass
        seq.notes.add(
            pitch=lh_bass,
            start_time=t,
            end_time=t + note_duration,
            velocity=65,
            part=lh_finger,
        )
        # RH arpeggio
        for j, (offset, finger) in enumerate(zip(rh_pattern, rh_fingers)):
            st = t + (j + 1) * note_duration
            seq.notes.add(
                pitch=rh_root + offset,
                start_time=st,
                end_time=st + note_duration,
                velocity=80,
                part=finger,
            )

    seq.total_time = 6 * 4 * note_duration
    seq.tempos.add(qpm=120)
    return midi_file.MidiFile(seq=seq)


def _d_major_arpeggio_one_hand(
    right_octave: int = 6,
    note_duration: float = 0.25,
) -> midi_file.MidiFile:
    """D major broken chord, right hand. D-F#-A-D up and down, 4 repetitions."""
    seq = music_pb2.NoteSequence()
    seq.sequence_metadata.title = "D major arpeggio (one hand)"
    seq.sequence_metadata.artist = "robopianist"

    root = 12 * right_octave + 2  # D6 = 74
    # D F# A D(up) A F# D(down)
    pattern = [0, 4, 7, 12, 7, 4, 0]
    fingering = [0, 1, 2, 4, 2, 1, 0]

    notes = pattern * 4
    fingers = fingering * 4

    for i, (pitch, finger) in enumerate(zip(notes, fingers)):
        seq.notes.add(
            pitch=root + pitch,
            start_time=i * note_duration,
            end_time=(i + 1) * note_duration,
            velocity=80,
            part=finger,
        )

    seq.total_time = len(notes) * note_duration
    seq.tempos.add(qpm=120)
    return midi_file.MidiFile(seq=seq)


def _d_major_arpeggio_two_hands(
    left_octave: int = 4,
    right_octave: int = 6,
    note_duration: float = 0.25,
) -> midi_file.MidiFile:
    """
    D major waltz-bass arpeggio (two hands), 6 cycles.
    LH plays D bass on beat 1; RH plays F#-A-D on beats 2-4.
    """
    seq = music_pb2.NoteSequence()
    seq.sequence_metadata.title = "D major arpeggio (two hands)"
    seq.sequence_metadata.artist = "robopianist"

    lh_bass = 12 * left_octave + 2    # D4 = 50
    rh_root = 12 * right_octave + 2   # D6 = 74
    rh_pattern = [4, 7, 12]           # F# A D
    rh_fingers = [1, 2, 4]
    lh_finger = 9

    for cycle in range(6):
        t = cycle * 4 * note_duration
        seq.notes.add(
            pitch=lh_bass,
            start_time=t,
            end_time=t + note_duration,
            velocity=65,
            part=lh_finger,
        )
        for j, (offset, finger) in enumerate(zip(rh_pattern, rh_fingers)):
            st = t + (j + 1) * note_duration
            seq.notes.add(
                pitch=rh_root + offset,
                start_time=st,
                end_time=st + note_duration,
                velocity=80,
                part=finger,
            )

    seq.total_time = 6 * 4 * note_duration
    seq.tempos.add(qpm=120)
    return midi_file.MidiFile(seq=seq)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ARPEGGIO_GENERATORS: Dict[str, Callable[[], midi_file.MidiFile]] = {
    "CMajorArpeggioOneHand": _c_major_arpeggio_one_hand,
    "CMajorArpeggioTwoHands": _c_major_arpeggio_two_hands,
    "DMajorArpeggioOneHand": _d_major_arpeggio_one_hand,
    "DMajorArpeggioTwoHands": _d_major_arpeggio_two_hands,
}

ARPEGGIO_ENVIRONMENT_NAMES = list(ARPEGGIO_GENERATORS.keys())


def make_arpeggio_env(
    name: str,
    seed: Optional[int],
    task_kwargs: dict,
    shift: int = 0,
) -> dm_env.Environment:
    """
    Instantiate a RoboPianist environment directly from an in-memory arpeggio
    MidiFile, bypassing save/load. This preserves the `part` (fingering) field
    so has_fingering() returns True and the observation space matches the target
    environment (1164 dims, fingering observable enabled).

    Equivalent to suite.load() but without the file roundtrip.
    """
    if name not in ARPEGGIO_GENERATORS:
        raise ValueError(
            f"Unknown arpeggio environment '{name}'. "
            f"Available: {list(ARPEGGIO_GENERATORS)}"
        )
    midi_obj = ARPEGGIO_GENERATORS[name]()
    if shift != 0:
        midi_obj = midi_obj.transpose(shift)
    return composer_utils.Environment(
        task=piano_with_shadow_hands.PianoWithShadowHands(midi=midi_obj, **task_kwargs),
        random_state=seed,
        strip_singleton_obs_buffer_dim=True,
        recompile_physics=False,
        legacy_step=True,
    )
