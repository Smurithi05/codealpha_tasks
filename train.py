"""
train.py
---------
This script:
  1. Loads the preprocessed notes (data/notes.pkl) created by preprocess.py.
  2. Converts the notes into numeric sequences that a neural network can understand.
  3. Builds an LSTM (Long Short-Term Memory) neural network.
  4. Trains the network to predict the "next note" given a sequence of previous notes.
  5. Saves the trained model and the note-to-number mappings to disk (in the 'model' folder).

THE AI PIPELINE (simple explanation for your evaluation):
    MIDI Dataset -> Preprocessing -> Note Sequences -> LSTM Training
    -> Next-Note Prediction -> Music Generation -> MIDI Output

This file is responsible for "LSTM Training" and "Next-Note Prediction".
generate.py is responsible for "Music Generation" and "MIDI Output".

HOW TO RUN:
    python train.py
"""

import os
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
DATA_FOLDER = "data"
NOTES_FILE = os.path.join(DATA_FOLDER, "notes.pkl")
MODEL_FOLDER = "model"
SEQUENCE_LENGTH = 50   # how many previous notes the model looks at to predict the next one


def load_notes():
    """Loads the list of notes/chords saved by preprocess.py."""
    if not os.path.exists(NOTES_FILE):
        raise FileNotFoundError(
            "notes.pkl not found. Please run 'python preprocess.py' first."
        )
    with open(NOTES_FILE, "rb") as f:
        notes = pickle.load(f)
    return notes


def prepare_sequences(notes, sequence_length=SEQUENCE_LENGTH):
    """
    Converts the raw note list into numeric input/output training pairs
    that the LSTM can learn from.

    CONCEPT (sliding window):
        Notes:            [C4, D4, E4, F4, G4, ...]
        Sequence 1 input:  [C4, D4, E4]   -> target output: F4
        Sequence 2 input:  [D4, E4, F4]   -> target output: G4
        ... and so on, sliding one step forward each time.

    The model's job during training is to learn: "given these previous
    notes, what note usually comes next?"
    """
    # Get every unique note/chord label in the dataset (sorted for consistency).
    pitch_names = sorted(set(notes))
    n_vocab = len(pitch_names)  # size of our "musical vocabulary"

    if n_vocab < 2:
        raise ValueError(
            "Not enough unique notes found to train on. "
            "Please add more/varied MIDI files to the data/ folder."
        )

    # Map each unique note/chord label to an integer, and back.
    note_to_int = {note_: number for number, note_ in enumerate(pitch_names)}

    # Reduce sequence length automatically if the dataset is very small,
    # so the project still works with tiny beginner datasets.
    effective_seq_len = min(sequence_length, max(2, len(notes) - 1))

    network_input = []
    network_output = []

    for i in range(len(notes) - effective_seq_len):
        seq_in = notes[i:i + effective_seq_len]
        seq_out = notes[i + effective_seq_len]
        network_input.append([note_to_int[n] for n in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)
    if n_patterns == 0:
        raise ValueError(
            "Could not create any training sequences. Your dataset is too small. "
            "Please add more MIDI files."
        )

    # Reshape input into the 3D format LSTM layers expect: (samples, time_steps, features)
    X = np.reshape(network_input, (n_patterns, effective_seq_len, 1))
    # Normalize values to the 0-1 range -- this helps the network train faster and more stably.
    X = X / float(n_vocab)

    # Store the target note as an integer instead of one-hot encoding.
    # This avoids creating a huge (n_patterns x n_vocab) array in RAM.
    y = np.asarray(network_output, dtype=np.int32)

    return X, y, pitch_names, note_to_int, n_vocab, effective_seq_len


def build_model(input_shape, n_vocab):
    """
    Builds a simple but effective LSTM network for music generation.

    ARCHITECTURE EXPLAINED SIMPLY:
      - LSTM layers: "remember" patterns across a sequence of notes, similar
        to how you remember the previous words in a sentence in order to
        guess a sensible next word.
      - Dropout layers: randomly switch off some neurons during training so
        the model doesn't just memorize the training songs (this reduces
        "overfitting" and helps it generalize / sound more musical).
      - Dense (fully connected) layer: turns the LSTM's internal understanding
        into one score per possible note in our vocabulary.
      - Softmax activation: converts those scores into probabilities (all
        add up to 1), so we can sample "the next note" from them.
    """
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(128))
    model.add(Dense(128))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model


def train(epochs=5, batch_size=64):
    """Main function: load data -> prepare sequences -> build model -> train -> save."""
    os.makedirs(MODEL_FOLDER, exist_ok=True)

    print("Loading notes...")
    notes = load_notes()

    print("Preparing training sequences...")
    X, y, pitch_names, note_to_int, n_vocab, seq_len = prepare_sequences(notes)
    print(f"Vocabulary size (unique notes/chords): {n_vocab}")
    print(f"Number of training sequences: {X.shape[0]}")
    print(f"Sequence length used: {seq_len}")

    # Keep batch size sensible for very small datasets.
    batch_size = max(1, min(batch_size, X.shape[0]))

    print("Building LSTM model...")
    model = build_model((X.shape[1], X.shape[2]), n_vocab)
    model.summary()

    # Save everything generate.py will need later (vocabulary + settings).
    with open(os.path.join(MODEL_FOLDER, "mappings.pkl"), "wb") as f:
        pickle.dump({
            "pitch_names": pitch_names,
            "note_to_int": note_to_int,
            "n_vocab": n_vocab,
            "sequence_length": seq_len
        }, f)

    # Automatically save the best version of the model seen during training.
    checkpoint_path = os.path.join(MODEL_FOLDER, "music_model.keras")
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )

    print("\nStarting training... (this may take a while on a normal laptop, especially without a GPU)")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])

    # Also explicitly save the final model, in case checkpointing missed the last epoch.
    model.save(checkpoint_path)
    print(f"\nTraining complete! Model saved to: {checkpoint_path}")


if __name__ == "__main__":
    # For a normal laptop without a GPU, keep epochs modest to start with.
    # Once you confirm everything works end-to-end, feel free to increase
    # epochs (e.g. 100-200) for better-sounding results.
    train(epochs=5, batch_size=64)
