"""
Generate structured curriculum MIDI files for RoboPianist training.
Produces scales and arpeggios ordered by difficulty.

Usage:
    python generate_curriculum.py --output_dir ./curriculum_midi
"""

import argparse
from pathlib import Path
import pretty_midi


# ── helpers ──────────────────────────────────────────────────────────────────

def make_midi(notes_left, notes_right, tempo=80, note_duration=0.4, gap=0.05):
    """
    Build a PrettyMIDI object from two lists of (pitch, start_time) tuples.
    notes_left / notes_right: list of MIDI pitch integers in order.
    Returns a PrettyMIDI object.
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    def add_track(note_list, program=0):
        instrument = pretty_midi.Instrument(program=program)
        for i, pitch in enumerate(note_list):
            start = i * (note_duration + gap)
            end = start + note_duration
            note = pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=end)
            instrument.notes.append(note)
        return instrument

    if notes_left:
        midi.instruments.append(add_track(notes_left))
    if notes_right:
        inst = pretty_midi.Instrument(program=0)
        for i, pitch in enumerate(notes_right):
            start = i * (note_duration + gap)
            end = start + note_duration
            note = pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=end)
            inst.notes.append(note)
        midi.instruments.append(inst)

    return midi


def scale_pitches(root, octave=4, ascending=True):
    """
    Generate a major scale from root note (e.g. 'C', 'G', 'F').
    Returns one octave of MIDI pitches, ascending then descending.
    """
    major_intervals = [0, 2, 4, 5, 7, 9, 11, 12]  # W W H W W W H
    note_names = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    root_midi = 12 * (octave + 1) + note_names[root]
    up = [root_midi + interval for interval in major_intervals]
    down = list(reversed(up[:-1]))  # descend back, don't repeat top
    return up + down


def arpeggio_pitches(root, octave=4):
    """
    Generate a broken chord (arpeggio) for a major triad.
    Goes up two octaves and back down.
    """
    note_names = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    root_midi = 12 * (octave + 1) + note_names[root]
    # Major triad intervals: root, major 3rd, perfect 5th, octave
    triad = [0, 4, 7, 12]
    up = [root_midi + i for i in triad] + [root_midi + 12 + i for i in triad]
    down = list(reversed(up[:-1]))
    return up + down


def hanon_pitches(root_midi, pattern_length=8):
    """
    Simplified Hanon-style finger exercise pattern.
    Ascending pattern: 1-3-4-5-4-3-2-1 style (relative semitones).
    Repeats starting from each scale degree.
    """
    pattern = [0, 4, 5, 7, 5, 4, 2, 0]  # relative semitone offsets
    notes = []
    for step in range(pattern_length):
        for offset in pattern:
            notes.append(root_midi + step + offset)
    return notes


# ── curriculum pieces ─────────────────────────────────────────────────────────

def generate_all(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    curriculum = []

    # ── Level 1: Single-hand scales ──────────────────────────────────────────
    for key in ['C', 'G', 'F']:
        pitches = scale_pitches(key, octave=4)

        # Right hand only
        midi = make_midi(notes_left=[], notes_right=pitches, tempo=80)
        fname = output_dir / f"scale_{key}_major_RH.mid"
        midi.write(str(fname))
        curriculum.append((fname.name, f"{key} major scale, right hand"))

        # Left hand only (one octave lower)
        pitches_lh = scale_pitches(key, octave=3)
        midi = make_midi(notes_left=pitches_lh, notes_right=[], tempo=80)
        fname = output_dir / f"scale_{key}_major_LH.mid"
        midi.write(str(fname))
        curriculum.append((fname.name, f"{key} major scale, left hand"))

    # ── Level 2: Hands together, scales ──────────────────────────────────────
    for key in ['C', 'G']:
        rh = scale_pitches(key, octave=4)
        lh = scale_pitches(key, octave=3)
        midi = make_midi(notes_left=lh, notes_right=rh, tempo=72)
        fname = output_dir / f"scale_{key}_major_HT.mid"
        midi.write(str(fname))
        curriculum.append((fname.name, f"{key} major scale, hands together"))

    # ── Level 3: Arpeggios ────────────────────────────────────────────────────
    for key in ['C', 'G', 'F']:
        pitches = arpeggio_pitches(key, octave=4)

        midi = make_midi(notes_left=[], notes_right=pitches, tempo=72)
        fname = output_dir / f"arpeggio_{key}_major_RH.mid"
        midi.write(str(fname))
        curriculum.append((fname.name, f"{key} major arpeggio, right hand"))

        pitches_lh = arpeggio_pitches(key, octave=3)
        midi = make_midi(notes_left=pitches_lh, notes_right=[], tempo=72)
        fname = output_dir / f"arpeggio_{key}_major_LH.mid"
        midi.write(str(fname))
        curriculum.append((fname.name, f"{key} major arpeggio, left hand"))

    # ── Level 4: Hands together arpeggios ────────────────────────────────────
    for key in ['C', 'G']:
        rh = arpeggio_pitches(key, octave=4)
        lh = arpeggio_pitches(key, octave=3)
        midi = make_midi(notes_left=lh, notes_right=rh, tempo=66)
        fname = output_dir / f"arpeggio_{key}_major_HT.mid"
        midi.write(str(fname))
        curriculum.append((fname.name, f"{key} major arpeggio, hands together"))

    # ── Level 5: Hanon-style finger exercises ─────────────────────────────────
    # C major root = MIDI 60
    hanon_rh = hanon_pitches(root_midi=60, pattern_length=5)
    midi = make_midi(notes_left=[], notes_right=hanon_rh, tempo=80,
                     note_duration=0.2, gap=0.02)
    fname = output_dir / f"hanon_C_RH.mid"
    midi.write(str(fname))
    curriculum.append((fname.name, "Hanon finger exercise, C, right hand"))

    # ── Print ordered curriculum ──────────────────────────────────────────────
    print(f"\nGenerated {len(curriculum)} curriculum MIDI files in: {output_dir}\n")
    print("Suggested training order:")
    for i, (fname, desc) in enumerate(curriculum, 1):
        print(f"  {i:2d}. {fname:<45} — {desc}")

    # Write curriculum order to a text file for reference
    order_file = output_dir / "curriculum_order.txt"
    with open(order_file, "w") as f:
        for fname, desc in curriculum:
            f.write(f"{fname}\t{desc}\n")
    print(f"\nOrder saved to: {order_file}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate curriculum MIDIs for RoboPianist")
    parser.add_argument("--output_dir", type=str, default="./curriculum_midi",
                        help="Directory to write MIDI files into")
    args = parser.parse_args()
    generate_all(Path(args.output_dir))