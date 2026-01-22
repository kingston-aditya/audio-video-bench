import torch
import os
import json
from torch.utils.data import Dataset, DataLoader

class video_questions_dataloader(Dataset):
    def __init__(self, data_path, options="abcd"):
        f = open(os.path.join(data_path, "coml_questions_prelim.json"))
        self.video_js = json.load(f)
        self.options = options

    def __len__(self):
        return len(self.video_js)

    def __getitem__(self, idx):
        # load the samples
        sample = self.video_js[idx]

        if self.options == "abcd":
            letters = ["B", "C", "D"]
            prompt = sample["quesion"][0] + "\n" + "A. " + sample["answer"][0] + "\n".join(f"{letter}. {value}" for letter, value in zip(letters, sample["option"].values()))
            answer = "A"
        elif self.options == "9qj4":
            letters = ["Q", "J", "4"]
            prompt = sample["quesion"][0] + "\n" + "9. " + sample["answer"][0] + "\n".join(f"{letter}. {value}" for letter, value in zip(letters, sample["option"].values()))
            answer = "9"
        elif self.options == "pqrs":
            letters = ["Q", "R", "S"]
            prompt = sample["quesion"][0] + "\n" + "P. " + sample["answer"][0] + "\n".join(f"{letter}. {value}" for letter, value in zip(letters, sample["option"].values()))
            answer = "P"
        elif self.options == "wqal":
            letters = ["Q", "A", "L"]
            prompt = sample["quesion"][0] + "\n" + "W. " + sample["answer"][0] + "\n".join(f"{letter}. {value}" for letter, value in zip(letters, sample["option"].values()))
            answer = "W"
        elif self.options == "bdca":
            letters = ["D", "C", "A"]
            prompt = sample["quesion"][0] + "\n" + "B. " + sample["answer"][0] + "\n".join(f"{letter}. {value}" for letter, value in zip(letters, sample["option"].values()))
            answer = "B"
        elif self.options == "explain":
            prompt = sample["quesion"][0]
            answer = "N"
        else:
            raise ValueError("This is wrong option.")

        return {"prompt": prompt, "answer": answer, "video": sample['video'][0]}

    @staticmethod
    def collate_fn(batch):
        prompt = [item["prompt"] for item in batch]
        answer = [item["answer"] for item in batch]
        video = [item["video"] for item in batch]
        return {"prompt": prompt, "answer": answer, "video":video}

