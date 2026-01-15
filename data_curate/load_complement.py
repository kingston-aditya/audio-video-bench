import torch
import os
import json
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(1, "/nfshomes/asarkar6/aditya/audio-video-bench/data_curate/")
from load_music_vqa import read_music_vqa

def load_complement(data_path):
    # open the original data
    original_js = read_music_vqa(data_path)

    # open the video questons file
    with open(os.path.join(data_path, "video_questions.json")) as f:
        video_js = json.load(f)
    
    return original_js, video_js

class video_questions_dataloader(Dataset):
    def __init__(self, data_path):
        self.original_js, self.video_js = load_complement(data_path)
        self.final_path = data_path

    def __len__(self):
        return len(self.video_js)

    def __getitem__(self, idx):
        # load the samples
        sample = self.video_js[idx]
        video_path = os.path.join(self.final_path, "MUCIS-AVQA-videos-Synthetic", self.original_js[idx]['video_id'])

        # load the videos
        question = sample["question"]
        answer = sample["answer"]
        return {"question": question, "answer": answer, "video":video_path}

    @staticmethod
    def collate_fn(batch):
        question = [item["question"] for item in batch]
        answer = [item["answer"] for item in batch]
        video = [item["video"] for item in batch]
        return {"question": question, "answer": answer, "video":video}

