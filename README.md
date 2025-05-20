# 🎧 EEG-based Personalized Music Generation using MusicGen

This project explores the generation of personalized music based on user EEG preferences using Meta's [MusicGen](https://github.com/facebookresearch/audiocraft). It consists of two major components:

- **Prompt generation & audio synthesis** based on EEG-derived genre preferences (`musicgentransformers.py`)

- **Optional fine-tuning of MusicGen** on custom audio datasets (`finetuning.py`)

## 💡 Features

- 🎵 Generates personalized audio tracks based on EEG-driven genre preferences

- 🧠 Categorizes music energy level (e.g., high-energy, calm, balanced)

- ✨ Dynamically crafts descriptive prompts using mood, genre, and cultural context

- 📊 Outputs:

- AI-generated `.wav` music

- Spectrogram of the audio

- Bar chart of genre preferences

## 📌 Notes

Used `facebook/musicgen-small` for lightweight inference

Music clips are short (~10–15s) due to GPU constraints (Colab)

Fine-tuning requires significant VRAM (recommended: A100)


