import torch
import os
import json
from torch.utils.data import Dataset, DataLoader

class video_questions_dataloader(Dataset):
    def __init__(self, data_path, options="abcd"):
        f = open(os.path.join(data_path, "supl_questions_prelim.json"))
        self.video_js = json.load(f)
        self.options = options

    def __len__(self):
        return len(self.video_js)

    def __getitem__(self, idx):
        # load the samples
        sample = self.video_js[idx]

        if self.options == "abcd":
            letters = ["B", "C", "D"]
            prompt = sample["quesion"] + "\n" + "A. " + sample["answer"] + "\n".join(f"{letter}. {value}" for letter, value in zip(letters, sample["option"].values()))
            answer = "A"
        elif self.options == "1234":
            letters = ["2", "3", "4"]
            prompt = sample["quesion"] + "\n" + "1. " + sample["answer"] + "\n".join(f"{letter}. {value}" for letter, value in zip(letters, sample["option"].values()))
            answer = "1"
        elif self.options == "pqrs":
            letters = ["Q", "R", "S"]
            prompt = sample["quesion"] + "\n" + "P. " + sample["answer"] + "\n".join(f"{letter}. {value}" for letter, value in zip(letters, sample["option"].values()))
            answer = "P"
        elif self.options == "wqal":
            letters = ["Q", "A", "L"]
            prompt = sample["quesion"] + "\n" + "W. " + sample["answer"] + "\n".join(f"{letter}. {value}" for letter, value in zip(letters, sample["option"].values()))
            answer = "W"
        else:
            raise ValueError("This is wrong option.")

        return {"prompt": prompt, "answer": answer, "video": sample['video'][0]}

    @staticmethod
    def collate_fn(batch):
        prompt = [item["prompt"] for item in batch]
        answer = [item["answer"] for item in batch]
        video = [item["video"] for item in batch]
        return {"prompt": prompt, "answer": answer, "video":video}

