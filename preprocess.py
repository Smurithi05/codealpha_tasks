"""
preprocess.py
--------------
This script reads all MIDI (.mid) files from the 'data' folder,
extracts the notes and chords from them using the music21 library,
and converts them into a format that can be used to train our
LSTM (Long Short-Term Memory) neural network.

WHAT IS A "NOTE SEQUENCE"?
A piece of music is basically a sequence of notes and chords played
one after another. We convert each note/chord into a simple text
label (for example "C4" for a single note, or "C4.E4.G4" for a chord
made of three notes played together), so the whole song becomes a
list of text labels -- similar to how a sentence is a list of words.
This lets us treat music generation like a "predict the next word"
problem, which is exactly what an LSTM is good at.

HOW TO RUN:
    python preprocess.py

OUTPUT:
This script creates one file inside the 'data' folder:
    notes.pkl -> the full list of note/chord labels from ALL songs,
                 combined one after another.
"""

import os
import pickle
from music21 import converter, instrument, note, chord

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
DATA_FOLDER = "data"                                   # folder containing your .mid files
NOTES_FILE = os.path.join(DATA_FOLDER, "notes.pkl")     # where we save the extracted notes


def get_notes():
    """
    Reads every MIDI file in the data/ folder and extracts a flat
    list of notes and chords (as strings) from all songs combined.

    Returns:
        notes (list of str): e.g. ['C4', 'E4.G4.C5', 'D4', ...]
    """
    notes = []

    # Find every MIDI file in the data folder (case-insensitive .mid/.midi)
    midi_files = [
        f for f in os.listdir(DATA_FOLDER)
        if f.lower().endswith(".mid") or f.lower().endswith(".midi")
    ]

    if not midi_files:
        raise FileNotFoundError(
            f"No MIDI files found in '{DATA_FOLDER}'. "
            "Please add some .mid files there before running preprocessing. "
            "See the README for where to download free MIDI files."
        )

    print(f"Found {len(midi_files)} MIDI file(s). Starting preprocessing...\n")

    for file_name in midi_files:
        file_path = os.path.join(DATA_FOLDER, file_name)
        print(f"Parsing: {file_name}")

        # Try to parse the MIDI file. Some files are corrupted or in an
        # unusual format, so we skip those instead of crashing the whole run.
        try:
            midi_stream = converter.parse(file_path)
        except Exception as e:
            print(f"  Could not parse {file_name}, skipping. Error: {e}")
            continue

        # A MIDI file can contain several "parts" (like separate instrument
        # tracks). We try to isolate a single instrument's part; if that
        # is not possible, we just use every note found in the file.
        try:
            parts = instrument.partitionByInstrument(midi_stream)
        except Exception:
            parts = None

        if parts:  # file has separate instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else:      # file has notes in one flat stream
            notes_to_parse = midi_stream.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                # A single note, e.g. "C4" (pitch name + octave)
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                # A chord = several notes played at the same time.
                # We store all of its pitches joined by dots, e.g. "C4.E4.G4"
                notes.append('.'.join(str(p) for p in element.pitches))

    print(f"\nExtracted {len(notes)} total notes/chords from all songs.")
    return notes


def save_notes(notes):
    """Saves the extracted notes list to disk using pickle, so train.py can load it later."""
    os.makedirs(DATA_FOLDER, exist_ok=True)
    with open(NOTES_FILE, "wb") as f:
        pickle.dump(notes, f)
    print(f"Saved notes to: {NOTES_FILE}")


if __name__ == "__main__":
    all_notes = get_notes()
    save_notes(all_notes)
    print("\nPreprocessing complete! You can now run: python train.py")
