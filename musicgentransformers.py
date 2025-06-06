# -*- coding: utf-8 -*-
"""MusicGenTransformers.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1DXOC1dE5Crl4D3xGy37tQPU1SxdH-67S
"""

!pip install --upgrade pip
!pip install --upgrade transformers scipy

!pip install torch

!pip install torchvision

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.io.wavfile as wavfile
from transformers import pipeline
from pathlib import Path
import os

def generate_dict(input_txt_path):
    known_genres = [
        "deep house", "indie", "electronics", "electronic dance",
        "new age", "ambient", "hindustani classical", "indian semi-classical",
        "indian folk", "soft jazz", "goth rock", "progressive"
    ]

    with open(input_txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    genre_dict = {}
    lines = text.splitlines()

    for line in lines:
        tokens = line.strip().split()
        if len(tokens) < 4 or not tokens[0].isdigit():
            continue
        try:
            song_no = int(tokens[0])
            line_lower = line.lower()
            matched = None
            for genre in known_genres:
                if genre in line_lower:
                    matched = genre
                    break
            if matched:
                genre_dict[song_no] = matched
        except:
            continue

    return genre_dict

def load_all_preferences(csv_path, genre_dict):
    df = pd.read_csv(csv_path)
    df['Genre1'] = df['TopGenre1'].map(genre_dict)
    df['Genre2'] = df['TopGenre2'].map(genre_dict)
    return df

def guess_mood(genre1, genre2):
    high_energy = {
        'deep house', 'indie', 'electronics', 'electronic dance',
        'goth rock', 'progressive'
    }
    calm = {
        'ambient', 'soft jazz', 'new age',
        'hindustani classical', 'indian semi-classical'
    }

    genres = {genre1.strip().lower(), genre2.strip().lower()}

    high_count = len(genres & high_energy)
    calm_count = len(genres & calm)

    if high_count > calm_count:
        return "high-energy"
    elif calm_count > high_count:
        return "relaxing"
    else:
        return "balanced"

def generate_prompt(genre1, genre2):
    mood = guess_mood(genre1, genre2)

    mood_map = {
        "high-energy": "an explosive and thrilling",
        "relaxing": "a dreamy and atmospheric",
        "balanced": "a smooth and evolving"
    }
    mood_desc = mood_map.get(mood, "a deep and emotional")

    vibe_map = {
        "deep house": "deep basslines with hypnotic grooves",
        "indie": "raw textures and nostalgic vibes",
        "electronics": "layered digital tones and sharp transitions",
        "electronic dance": "fast-paced rhythms and club energy",
        "new age": "soft pads and spiritual resonance",
        "ambient": "floating textures and calm atmosphere",
        "hindustani classical": "traditional ragas with meditative flow",
        "indian semi-classical": "blended folk-classical melodies",
        "indian folk": "earthy traditional rhythms and cultural depth",
        "soft jazz": "warm brass tones and late-night groove",
        "goth rock": "dark guitars and moody ambiance",
        "progressive": "complex instrumental builds and expressive flow"
    }

    vibe1 = vibe_map.get(genre1, f"elements of {genre1}")
    vibe2 = vibe_map.get(genre2, f"elements of {genre2}")

    scenario = "for a surreal sonic experience tailored to a unique soul"

    return f"Create {mood_desc} track blending {vibe1} and {vibe2}, {scenario}."

def generate_music(prompt, user_id, synthesiser, output_dir):
    print(f"\n🎧 Generating music for User {user_id}...")
    print(f"Prompt: {prompt}\n")

    music = synthesiser(prompt, forward_params={"do_sample": True, "max_new_tokens": 1024})

    audio = music["audio"]
    sr = music["sampling_rate"]
    file_path = output_dir / "music.wav"
    wavfile.write(str(file_path), rate=sr, data=audio)
    return file_path

def plot_spectrogram(file_path, user_id, output_dir):
    y, sr = librosa.load(file_path)
    plt.figure(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram for User {user_id}')
    plt.tight_layout()
    out_path = output_dir / "spectrogram.png"
    plt.savefig(out_path)
    plt.close()
    print(f"→ Spectrogram saved: {out_path}")

def process_all_users(csv_path, json_path):
    genre_dict = generate_dict(json_path)
    df = load_all_preferences(csv_path, genre_dict)

    synthesiser = pipeline("text-to-audio", "facebook/musicgen-small")

    for _, row in df.iterrows():
        user_id = row['Subject']
        genre1 = row['Genre1']
        genre2 = row['Genre2']

        output_dir = Path(f"user_{user_id}")
        output_dir.mkdir(exist_ok=True)

        prompt = generate_prompt(genre1, genre2)
        audio_path = generate_music(prompt, user_id, synthesiser, output_dir)
        plot_spectrogram(audio_path, user_id, output_dir)

csv_path = "MusicGen_Prompts.csv"
json_path = "Song_Description"
process_all_users(csv_path, json_path)

