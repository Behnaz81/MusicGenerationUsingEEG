# !git clone https://github.com/facebookresearch/audiocraft.git
# !pip install -r requirements.txt
# !pip install -e .

from audiocraft.models import MusicGen
from audiocraft.training.trainer import Trainer
from audiocraft.data.audio_dataset import AudioDataset
import shutil
import os
from google.colab import files

csv_path = "/content/musicgen_training_metadata.csv"  
audio_folder = "/content/audio"  
output_dir = "/content/musicgen_finetuned" 

config = {
    "lr": 2e-5,
    "batch_size": 2,
    "max_steps": 500,
    "output_dir": output_dir,
    "sample_rate": 32000,
    "duration": 10,
    "model": "medium"
}

dataset = AudioDataset(
    csv_path=csv_path,
    audio_folder=audio_folder,
    sample_rate=config["sample_rate"],
    duration=config["duration"]
)

model = MusicGen.get_pretrained(config["model"])

trainer = Trainer(
    model=model,
    dataset=dataset,
    batch_size=config["batch_size"],
    lr=config["lr"],
    max_steps=config["max_steps"],
    save_path=config["output_dir"]
)

trainer.train()

shutil.make_archive(output_dir, 'zip', output_dir)

files.download(output_dir + ".zip")