# !pip install --upgrade pip
# !pip install --upgrade transformers scipy

# !pip install torch

# !pip install torchvision

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.io.wavfile as wavfile
from transformers import pipeline
from pathlib import Path
import os

def load_all_preferences(csv_path):
    df = pd.read_csv(csv_path)
    df['Preference (%)'] = df['Preference (%)'].astype(float)
    df = df[df['Preference (%)'] > 0]
    df['Genre'] = df['Genre'].str.strip().str.lower()
    return df

def guess_mood(preferences_df):
    high_energy = [
        'deep house', 'indie', 'electronics', 'electronic dance',
        'goth rock', 'progressive instrumental rock'
    ]
    calm = [
        'ambient', 'soft jazz', 'new age',
        'hindustani classical', 'indian semi-classical'
    ]

    preferences_df = preferences_df.copy()
    preferences_df['energy'] = preferences_df['Genre'].str.strip().str.lower().apply(
        lambda g: 'high' if g in high_energy else ('calm' if g in calm else 'neutral')
    )

    high = preferences_df[preferences_df['energy'] == 'high']['Preference (%)'].sum()
    calm_sum = preferences_df[preferences_df['energy'] == 'calm']['Preference (%)'].sum()
    neutral = preferences_df[preferences_df['energy'] == 'neutral']['Preference (%)'].sum()

    if high >= max(calm_sum, neutral):
        return "high-energy"
    elif calm_sum >= max(high, neutral):
        return "relaxing"
    else:
        return "balanced"

def generate_prompt(preferences_df):
    weighted_genres = preferences_df.groupby('Genre')['Preference (%)'].sum().sort_values(ascending=False)
    top_genres = weighted_genres.head(3).index.tolist()
    mood = guess_mood(preferences_df)

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
        "progressive instrumental rock": "complex instrumental builds and expressive flow"
    }

    genre_vibes = [vibe_map.get(g, f"elements of {g}") for g in top_genres]
    vibe_string = ", ".join(genre_vibes)

    if "indian folk" in top_genres and "indian semi-classical" in top_genres:
        scenario = "to echo the colors and emotions of a cultural celebration"
    elif "ambient" in top_genres and "new age" in top_genres:
        scenario = "for peaceful introspection during a quiet evening"
    elif "goth rock" in top_genres and "electronic dance" in top_genres:
        scenario = "to fuel a mysterious midnight rave"
    elif "soft jazz" in top_genres and "hindustani classical" in top_genres:
        scenario = "to accompany deep thoughts in a candle-lit courtyard"
    else:
        scenario = "for a unique moment of emotional expression"

    prompt = (
        f"Create {mood_desc} track blending {vibe_string}, {scenario}."
        " The piece should feel tailored to the listener's emotional landscape."
    )

    return prompt, mood

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

def plot_preferences_bar(preferences_df, user_id, output_dir):
    preferences_df = preferences_df.groupby("Genre")["Preference (%)"].sum().sort_values()
    preferences_df.plot(kind='barh', title=f'User {user_id} Preferences')
    plt.tight_layout()
    out_path = output_dir / "preferences.png"
    plt.savefig(out_path)
    plt.close()
    print(f"→ Preferences bar chart saved: {out_path}")

def process_all_users(csv_path):
    df = load_all_preferences(csv_path)
    synthesiser = pipeline("text-to-audio", "facebook/musicgen-small")

    for user_id, user_df in df.groupby("UserID"):
        output_dir = Path(f"user_{user_id}")
        output_dir.mkdir(exist_ok=True)

        prompt, mood = generate_prompt(user_df)
        audio_path = generate_music(prompt, user_id, synthesiser, output_dir)
        spec_path = plot_spectrogram(audio_path, user_id, output_dir)
        bar_path = plot_preferences_bar(user_df, user_id, output_dir)

csv_path = "mock_music_preferences.csv"
process_all_users(csv_path)

