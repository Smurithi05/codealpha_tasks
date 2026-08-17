"""
generate.py
------------
This script loads the trained LSTM model and uses it to generate a
brand-new sequence of notes, note by note, then converts that sequence
into a real MIDI file you can open and play.

HOW GENERATION WORKS (simple explanation):
  1. We pick a random short sequence from the training data as a "seed"
     -- a starting point, so the model has some musical context to continue.
  2. We ask the model: "Given these previous notes, what note comes next?"
  3. The model outputs a probability for every possible note in its
     vocabulary; we randomly pick one, weighted by those probabilities
     (this is controlled by the 'temperature' setting below).
  4. We add the chosen note to our sequence, drop the oldest note from
     the window, and repeat -- generating one note at a time.
  5. Finally, we convert the full generated sequence of note/chord labels
     back into music21 Note/Chord objects and save them as a MIDI file.

HOW TO RUN:
    python generate.py
"""

import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from music21 import stream, note, chord, instrument

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
MODEL_FOLDER = "model"
MODEL_PATH = os.path.join(MODEL_FOLDER, "music_model.keras")
MAPPINGS_PATH = os.path.join(MODEL_FOLDER, "mappings.pkl")
DATA_FOLDER = "data"
NOTES_FILE = os.path.join(DATA_FOLDER, "notes.pkl")
GENERATED_FOLDER = "generated"
OUTPUT_FILE = os.path.join(GENERATED_FOLDER, "generated_music.mid")


def load_resources():
    """Loads the trained model, the note mappings, and the original notes (used for seeding)."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained model not found. Please run 'python train.py' first.")
    if not os.path.exists(MAPPINGS_PATH):
        raise FileNotFoundError("mappings.pkl not found. Please run 'python train.py' first.")
    if not os.path.exists(NOTES_FILE):
        raise FileNotFoundError("notes.pkl not found. Please run 'python preprocess.py' first.")

    model = load_model(MODEL_PATH)

    with open(MAPPINGS_PATH, "rb") as f:
        mappings = pickle.load(f)

    with open(NOTES_FILE, "rb") as f:
        notes = pickle.load(f)

    return model, mappings, notes


def generate_notes(model, mappings, notes, num_notes=200, temperature=1.0):
    """
    Generates a new sequence of note/chord labels using the trained model.

    Args:
        num_notes (int): how many notes/chords to generate.
        temperature (float): controls randomness/creativity of the output.
            - Lower (e.g. 0.5)  -> more predictable, repetitive, "safe" music.
            - 1.0               -> follows the model's learned probabilities as-is.
            - Higher (e.g. 1.2+)-> more random/experimental (can sound less coherent).
    """
    pitch_names = mappings["pitch_names"]
    note_to_int = mappings["note_to_int"]
    n_vocab = mappings["n_vocab"]
    sequence_length = mappings["sequence_length"]
    int_to_note = {number: note_ for note_, number in note_to_int.items()}

    # Rebuild all possible starting windows from the original training notes,
    # so we can pick a realistic random seed to begin generating from.
    network_input = []
    for i in range(len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        network_input.append([note_to_int[n] for n in seq_in])

    if not network_input:
        raise ValueError("Not enough notes available to build a seed sequence.")

    start_index = np.random.randint(0, len(network_input))
    pattern = list(network_input[start_index])

    prediction_output = []

    print(f"Generating {num_notes} notes...")
    for step in range(num_notes):
        # Reshape and normalize the current pattern exactly like during training.
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        # Ask the model for probabilities over every possible next note.
        prediction = model.predict(prediction_input, verbose=0)[0]

        # Apply "temperature" to make choices more/less random.
        prediction = np.log(prediction + 1e-8) / temperature
        exp_preds = np.exp(prediction)
        probabilities = exp_preds / np.sum(exp_preds)

        # Randomly sample the next note, weighted by the model's probabilities.
        index = np.random.choice(len(probabilities), p=probabilities)

        result = int_to_note[index]
        prediction_output.append(result)

        # Slide the window forward: add the new note, drop the oldest one.
        pattern.append(index)
        pattern = pattern[1:]

        if (step + 1) % 50 == 0:
            print(f"  ...generated {step + 1}/{num_notes} notes")

    return prediction_output


def create_midi(prediction_output, output_path=OUTPUT_FILE):
    """
    Converts a list of note/chord label strings back into a music21 Stream
    and saves it as a real, playable MIDI file.
    """
    offset = 0  # tracks the position in time (in quarter-note lengths)
    output_notes = []

    for pattern in prediction_output:
        if '.' in pattern:
            # This label represents a CHORD, e.g. "C4.E4.G4"
            chord_pitches = pattern.split('.')
            notes_in_chord = []
            for pitch_str in chord_pitches:
                new_note = note.Note(pitch_str)
                new_note.storedInstrument = instrument.Piano()
                notes_in_chord.append(new_note)
            new_chord = chord.Chord(notes_in_chord)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            # This label represents a SINGLE NOTE, e.g. "C4"
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # Move forward in time before placing the next note/chord.
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    os.makedirs(GENERATED_FOLDER, exist_ok=True)
    midi_stream.write('midi', fp=output_path)
    print(f"\nGenerated MIDI file saved to: {output_path}")


if __name__ == "__main__":
    model, mappings, notes = load_resources()
    generated_notes = generate_notes(model, mappings, notes, num_notes=200, temperature=1.0)
    create_midi(generated_notes)
    print("Done! Open the generated file in any MIDI player, DAW, or notation app "
          "(e.g. Windows Media Player, VLC, MuseScore, GarageBand) to listen to it.")
