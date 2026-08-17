"""
app.py
-------
A simple, beginner-friendly command-line menu that ties the whole
project together: preprocessing, training, and generating music,
all from one place.

This is entirely optional -- you can also just run preprocess.py,
train.py, and generate.py directly, one after another, from the
terminal instead of using this menu.

HOW TO RUN:
    python app.py
"""

import os
import sys


def run_preprocess():
    print("\n--- Running preprocess.py ---")
    os.system(f'"{sys.executable}" preprocess.py')


def run_train():
    print("\n--- Running train.py ---")
    os.system(f'"{sys.executable}" train.py')


def run_generate():
    print("\n--- Running generate.py ---")
    os.system(f'"{sys.executable}" generate.py')


def main():
    while True:
        print("\n===== AI Music Generation - Main Menu =====")
        print("1. Preprocess MIDI dataset   (data/*.mid -> data/notes.pkl)")
        print("2. Train LSTM model          (data/notes.pkl -> model/music_model.keras)")
        print("3. Generate new music        (-> generated/generated_music.mid)")
        print("4. Run all steps in order (1 -> 2 -> 3)")
        print("5. Exit")
        choice = input("Enter your choice (1-5): ").strip()

        if choice == "1":
            run_preprocess()
        elif choice == "2":
            run_train()
        elif choice == "3":
            run_generate()
        elif choice == "4":
            run_preprocess()
            run_train()
            run_generate()
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number from 1 to 5.")


if __name__ == "__main__":
    main()
