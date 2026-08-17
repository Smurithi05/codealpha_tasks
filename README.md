# 🎵 AI Music Generation using LSTM

**CodeAlpha AI Internship — Task 3: Music Generation with AI**

## 1. Project Title

**AI Music Generation using an LSTM Neural Network**

## 2. Objective

To build an AI system that learns musical patterns (notes, chords, and their
order) from a dataset of MIDI songs, and then **generates a brand-new piece
of original music** in a similar style, saved as a playable `.mid` file.

## 3. How the System Works (Simple Explanation for your Faculty)

Think of music generation the same way a phone keyboard predicts your next
word while typing. Instead of words, our model learns to predict the
**next musical note**, based on the notes that came before it.

```
MIDI Dataset → Preprocessing → Note Sequences → LSTM Training
→ Next-Note Prediction → Music Generation → MIDI Output
```

| Stage | What happens |
|---|---|
| **MIDI Dataset** | We start with real songs saved as `.mid` files. |
| **Preprocessing** | We read every song and extract its notes and chords, in order, as simple text labels (e.g. `C4`, `E4.G4.C5`). |
| **Note Sequences** | We slide a "window" of, say, 50 notes across each song to create thousands of `(previous 50 notes → next note)` training examples. |
| **LSTM Training** | An LSTM (a neural network built for sequences) learns the statistical patterns of which notes tend to follow which. |
| **Next-Note Prediction** | Once trained, given any sequence of notes, the model can predict a probability for what note comes next. |
| **Music Generation** | We seed the model with a starting sequence, ask it to predict the next note, add that note to the sequence, and repeat hundreds of times — generating a brand new song note-by-note. |
| **MIDI Output** | The generated sequence of notes/chords is converted back into a real `.mid` file you can play in any music player. |

## 4. Dataset

This project needs a small collection of **MIDI (`.mid`) files** placed
inside the `data/` folder. MIDI files store music as *notes and timing*
(not audio), which is exactly what our LSTM needs to learn from.

**Free, legal sources for MIDI files:**

- **Classical Archives / Kunst der Fuge (Bach)** — https://www.kunstderfuge.com (free samples)
- **Classical Piano MIDI Page** — http://www.piano-midi.de (free, classical piano pieces — great beginner dataset)
- **freemidi.org** — https://freemidi.org (large free collection across genres)
- **MuseScore.com** — https://musescore.com (many user-uploaded scores can be exported/downloaded as MIDI)
- **Lakh MIDI Dataset (LMD)** — https://colinraffel.com/projects/lmd/ (large, ~176,000 MIDI files; use only a small subset for a laptop)

**For a beginner/laptop-friendly project:**
- Download **10–30 MIDI files** of a similar style (e.g. all classical piano,
  or all from one composer). A smaller, *consistent* dataset trains faster
  and produces more coherent results than a huge, mixed one.
- Place the `.mid` files directly inside the `data/` folder:
  ```
  music-generation-ai/data/song1.mid
  music-generation-ai/data/song2.mid
  ...
  ```
- If you just want to **test that the project works** before downloading a
  real dataset, see the "Quick Test with Fake Data" section below — you can
  generate a few tiny sample MIDI files with 3 lines of Python.

## 5. Preprocessing (`preprocess.py`)

- Reads every `.mid` file in `data/` using the **music21** library.
- Walks through each song and extracts every `Note` and `Chord`:
  - A single note becomes a text label like `"C4"`.
  - A chord (multiple notes played together) becomes a label like
    `"C4.E4.G4"` (pitches joined by dots).
- All notes from all songs are combined into **one long list**, saved to
  `data/notes.pkl` (a Python pickle file) for the training step to use.

## 6. LSTM Architecture (`train.py`)

```
Input (sequence of 50 previous notes, normalized)
   ↓
LSTM(128 units, return_sequences=True)   ← learns short & long-term note patterns
   ↓
Dropout(0.3)                             ← prevents overfitting
   ↓
LSTM(128 units)                          ← further refines the sequence understanding
   ↓
Dense(128)                               ← fully connected layer
   ↓
Dropout(0.3)
   ↓
BatchNormalization()                     ← stabilizes/speeds up training
   ↓
Dense(n_vocab)                           ← one output per possible note in the dataset
   ↓
Softmax activation                       ← converts outputs into probabilities
```

- **Why LSTM?** A plain neural network has no memory of order. Music is
  fundamentally about *sequence* (what note came before matters a lot).
  LSTM (Long Short-Term Memory) is a type of RNN specifically designed to
  remember useful information across a sequence, which is exactly what's
  needed to model melodies.
- **Loss function:** `categorical_crossentropy` (standard for multi-class
  "which note is next" classification).
- **Optimizer:** `adam` (a well-performing default choice for most projects).

## 7. Training (`train.py`)

- Loads `data/notes.pkl`.
- Builds `(input_sequence → next_note)` training pairs using a sliding
  window of 50 notes.
- Normalizes inputs and one-hot encodes outputs.
- Trains the LSTM for a number of epochs (default: 50 — enough for a
  small laptop dataset; increase later for better results).
- Saves:
  - The trained model → `model/music_model.keras`
  - The note vocabulary/mappings → `model/mappings.pkl`

## 8. Music Generation (`generate.py`)

- Loads the trained model and mappings.
- Picks a random seed sequence from the original notes to start from.
- Repeatedly:
  1. Feeds the current sequence into the model.
  2. Gets a probability distribution over all possible next notes.
  3. Randomly samples the next note using those probabilities (controlled
     by a `temperature` setting for more/less randomness).
  4. Adds the new note to the sequence and slides the window forward.
- Repeats this 200 times by default to generate a new song.

## 9. Output

- The generated note sequence is converted back into `music21` `Note` /
  `Chord` objects and written to:
  ```
  generated/generated_music.mid
  ```
- This is a standard MIDI file you can open with:
  - Windows Media Player / VLC
  - MuseScore (free, also lets you view it as sheet music)
  - Any Digital Audio Workstation (GarageBand, FL Studio, Ableton, etc.)

## 10. Technologies Used

| Tool | Purpose |
|---|---|
| **Python 3** | Programming language |
| **TensorFlow / Keras** | Building and training the LSTM neural network |
| **music21** | Parsing MIDI files and writing generated MIDI output |
| **NumPy** | Numeric array handling for model input/output |

All libraries used are **free and open-source** — no paid APIs are required.

---

## 11. Installation (Windows)

### Step 1 — Install Python
Make sure you have **Python 3.9–3.11** installed (recommended for
TensorFlow compatibility). Check with:
```
python --version
```
If not installed, download it from https://www.python.org/downloads/
(tick **"Add Python to PATH"** during installation).

### Step 2 — Open the project folder in terminal
Open **Command Prompt** or **PowerShell**, then navigate to the project:
```
cd path\to\music-generation-ai
```

### Step 3 — (Recommended) Create a virtual environment
```
python -m venv venv
venv\Scripts\activate
```
Your terminal prompt should now show `(venv)` at the start.

### Step 4 — Install dependencies
```
pip install -r requirements.txt
```
This installs TensorFlow, music21, and NumPy. It may take a few minutes.

---

## 12. How to Run — Step by Step

### Step 1 — Add MIDI files to the dataset
Place several `.mid` files into the `data/` folder (see Section 4 for
free download sources).

**Quick test with fake data** (optional, to confirm everything works
before downloading a real dataset): run this once from the project folder
with your virtual environment active:
```
python -c "from music21 import stream, note, chord; import random; random.seed(1); [stream.Stream([note.Note(random.choice(['C4','D4','E4','F4','G4','A4','B4'])) for _ in range(80)]).write('midi', fp=f'data/test_{i}.mid') for i in range(3)]"
```
This creates 3 tiny random MIDI files in `data/` just so you can test the
full pipeline runs correctly end-to-end.

### Step 2 — Preprocess the dataset
```
python preprocess.py
```
This creates `data/notes.pkl`.

### Step 3 — Train the LSTM model
```
python train.py
```
This trains the model and saves it to `model/music_model.keras` (plus
`model/mappings.pkl`). On a normal laptop (no GPU) with a small dataset
(10–30 short songs), this typically takes **anywhere from a few minutes
to an hour**, depending on dataset size and number of epochs. You can
lower `epochs` in `train.py`'s `if __name__ == "__main__":` block for a
faster (lower quality) test run, e.g. `train(epochs=10)`.

### Step 4 — Generate new music
```
python generate.py
```
This creates `generated/generated_music.mid`. Open it in any MIDI player
to listen!

### (Optional) Use the all-in-one menu
Instead of running the three scripts separately, you can run:
```
python app.py
```
and choose options from a simple menu (preprocess / train / generate / run
all steps).

---

## 13. Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'tensorflow'` | Run `pip install -r requirements.txt` again, and make sure your virtual environment is activated (`venv\Scripts\activate`). |
| `pip install` fails / TensorFlow install error | Make sure you're using a 64-bit Python 3.9–3.11 installation. TensorFlow does not support 32-bit Python or very new Python versions (e.g. 3.13) on Windows. |
| `FileNotFoundError: No MIDI files found in 'data'` | You forgot to add `.mid` files to the `data/` folder — see Step 1. |
| `FileNotFoundError: notes.pkl not found` | Run `python preprocess.py` before `python train.py`. |
| `FileNotFoundError: Trained model not found` | Run `python train.py` before `python generate.py`. |
| Training is very slow | Reduce `epochs` in `train.py`, reduce the number/length of MIDI files in `data/`, or reduce `SEQUENCE_LENGTH`. A GPU is **not required** for this project, just slower on CPU. |
| Generated music sounds repetitive/random | This is normal for a small dataset or few training epochs. Try: more/varied training MIDI files, more epochs, or adjusting `temperature` in `generate.py` (lower = more predictable, higher = more random). |
| `music21` can't find a MIDI player when trying `.show('midi')` | You don't need this — the project only *writes* MIDI files with `.write('midi', ...)`, which doesn't require a configured player. Just open the generated `.mid` file with VLC, Windows Media Player, or MuseScore afterward. |
| Errors parsing a specific MIDI file | `preprocess.py` automatically skips files it can't parse and continues with the rest — check the printed warning for which file to remove or replace. |
| `UnicodeDecodeError` or weird filename issues | Avoid special characters/spaces in your MIDI filenames; rename them to simple names like `song1.mid`, `song2.mid`. |

---

## 14. Project Structure

```
music-generation-ai/
│
├── data/                      # Put your training .mid files here
│   └── notes.pkl              # (created by preprocess.py)
│
├── generated/
│   └── generated_music.mid    # (created by generate.py)
│
├── model/
│   ├── music_model.keras      # (created by train.py) trained LSTM model
│   └── mappings.pkl           # (created by train.py) note vocabulary
│
├── preprocess.py               # Step 1: MIDI -> note sequences
├── train.py                    # Step 2: build & train the LSTM
├── generate.py                 # Step 3: generate new music -> MIDI
├── app.py                      # Optional: simple menu to run everything
├── requirements.txt
└── README.md
```

## 15. Summary (for your evaluation presentation)

> "I collected free MIDI songs, converted each song into a sequence of
> note/chord labels using music21, then trained an LSTM neural network to
> predict the next note given the previous 50 notes — similar to
> predictive text, but for music. After training, I generate new music by
> repeatedly asking the model to predict the next note starting from a
> random seed, and I convert the resulting sequence back into a real MIDI
> file using music21."
