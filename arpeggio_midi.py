from pathlib import Path
from typing import Callable, Dict, Optional

import dm_env
from note_seq import music_pb2
from robopianist.music import midi_file
from robopianist.suite.tasks import piano_with_shadow_hands
from mujoco_utils import composer_utils


def _c_major_arpeggio_one_hand(
    right_octave: int = 6,
    note_duration: float = 0.25,
) -> midi_file.MidiFile:
    """C major broken chord, right hand. C-E-G-C up and down, 4 repetitions."""
    seq = music_pb2.NoteSequence()
    seq.sequence_metadata.title = "C major arpeggio (one hand)"
    seq.sequence_metadata.artist = "robopianist"

    root = 12 * right_octave
    pattern = [0, 4, 7, 12, 7, 4, 0] # C E G C G E C
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
    
    seq = music_pb2.NoteSequence()
    seq.sequence_metadata.title = "C major arpeggio (two hands)"
    seq.sequence_metadata.artist = "robopianist"

    lh_bass = 12 * left_octave 
    rh_root = 12 * right_octave
    rh_pattern = [4, 7, 12] # E G C 
    rh_fingers = [1, 2, 4]
    lh_finger = 9 

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
    
    seq = music_pb2.NoteSequence()
    seq.sequence_metadata.title = "D major arpeggio (one hand)"
    seq.sequence_metadata.artist = "robopianist"

    root = 12 * right_octave + 2 
    pattern = [0, 4, 7, 12, 7, 4, 0] # D F# A D(up) A F# D(down)
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
    
    seq = music_pb2.NoteSequence()
    seq.sequence_metadata.title = "D major arpeggio (two hands)"
    seq.sequence_metadata.artist = "robopianist"

    lh_bass = 12 * left_octave + 2  
    rh_root = 12 * right_octave + 2 
    rh_pattern = [4, 7, 12]
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
