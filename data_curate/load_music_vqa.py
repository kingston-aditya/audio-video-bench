import datasets
import zipfile
import json
import os

import torch
from torch.utils.data import Dataset, DataLoader

import pdb

def read_music_vqa(data_path):
    # open the file
    with open(os.path.join(data_path, "avqa-train_fake.json")) as f:
        js = json.load(f)
        for idx, item in enumerate(js):
            js[idx]['video_id'] = os.path.join(data_path, "MUCIS-AVQA-videos-Synthetic", js[idx]['video_id']+".mp4")

    # return json file
    return js

class music_vqa_dataloader(Dataset):
    def __init__(self, data_path):
        self.data = read_music_vqa(data_path)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @staticmethod
    def collate_fn(batch):
        video = [item["video_id"] for item in batch]
        prompt = [item["question_content"] for item in batch]
        answer = [item["anser"] for item in batch]
        return {"video":video, "prompt": prompt, "answer": answer}

class contra_vqa_dataloader(Dataset):
    def __init__(self, data_path):
        self.data = read_music_vqa(data_path)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @staticmethod
    def collate_fn(batch):
        video = [item["video_id"] for item in batch]
        prompt = [item["question_content"] for item in batch]
        answer = [item["anser"] for item in batch]
        return {"video":video, "prompt": prompt, "answer": answer}
        

# if __name__ == "__main__":
#     data = music_vqa_dataloader("/nfshomes/asarkar6/trinity/music-vqa/")
#     dataloader = DataLoader(data, shuffle=True, batch_size=1, collate_fn=data.collate_fn)
#     for idx, batch in enumerate(dataloader):
#         pdb.set_trace()




