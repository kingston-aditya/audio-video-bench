from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import torch
import zipfile
from decord import VideoReader, cpu, gpu
from PIL import Image
import numpy as np
import os
import re

import json
import sys
sys.path.insert(1, "/nfshomes/asarkar6/aditya/audio-video-bench/")
from data_curate.load_music_vqa import music_vqa_dataloader

import argparse
from tqdm import tqdm

import pdb

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Inference script.")
    parser.add_argument(
        "--pretrained_lmm_name",
        type=str,
        default="Qwen/Qwen2-Audio-7B-Instruct",
        help="Path to LMM.",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="data directory",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="",
        help="cache directory",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="cache directory",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Number of frames to be sampled.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

# create conversation for unimodal and multimodal prompts
def create_questions(path):
    prompt1 = "You are given a video as input. Your task is to generate multiple-choice questions (MCQs) based strictly on the visual content, actions, events, or setting in the video."
    prompt2 = "Rules and Constraints: \n 1) Do NOT create questions about musical instruments, music theory, songs, or audio-specific musical elements. 2) Each question must be answerable from the video alone. 3) Avoid trivial or overly generic questions. 4) Focus on observable facts, reasoning, or interpretation (e.g., actions, intent, cause-effect, environment, sequence)."
    prompt3 = "MCQ requirements: \n For each question: 1) Provide exactly 4 options, labeled A, B, C, D. 2) Only one option must be correct 3) The incorrect options should be plausible but clearly wrong. 4) Clearly indicate the correct answer."
    prompt4 = "For each question, use the following format: \n Question: \n Options: A. \n B. \n C. \n D. \n Correct Answer: <A/B/C/D>"

    # load all modalities
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": path,
                    "max_pixels": 360 * 420,
                    "max_frames": 32,
                },
                {"type": "text", "text": prompt1 + "\n" +  prompt2 + "\n" + prompt3 + "\n" + prompt4},
            ],
        }
    ]
        
    return conversation

def process_videos(zip_path, video_path, num_frames=32):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        mini_file_name = "/".join(video_path[0].split("/")[-2:])
        if mini_file_name in zip_ref.namelist():
            with zip_ref.open(mini_file_name) as file:
                vr = VideoReader(file, ctx=cpu(0))
        else:
            print(f"{mini_file_name} is missing!")

    frame_indices = [int(i * len(vr) / 32) for i in range(num_frames)]
    sampled_frames = [Image.fromarray(vr[i].asnumpy()[...,::-1]).resize((224, 224), Image.BILINEAR) for i in frame_indices]
    return sampled_frames

def inference(video):
    # create the conversation
    conversations = create_questions(video)

    # do the inference
    text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    images, videos, video_kwargs = process_vision_info([conversations], return_video_kwargs=True)

    # compute the inputs
    inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt", padding=True, **video_kwargs,)
    inputs = inputs.to(model.device).to(model.dtype)

    # get the output text
    with torch.no_grad():
        text_ids = model.generate(**inputs, max_new_tokens=128)
        text_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, text_ids)
        ]
        response = processor.batch_decode(text_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    torch.cuda.empty_cache()
    return response[0]

if __name__ == "__main__":
    args = parse_args()
    model_path = args.pretrained_lmm_name

    zip_path = os.path.join(args.data_dir, "music-avqa-synthetic.zip")

    f = open(os.path.join(args.data_dir, "supl_questions_prelim.json"), "w")
    
    # load dataset
    precomputed_dataset = music_vqa_dataloader(args.data_dir)
    train_dataloader = torch.utils.data.DataLoader(
        precomputed_dataset,
        shuffle=False,
        collate_fn=precomputed_dataset.collate_fn,
        batch_size=args.batch_size,
        num_workers=1,
    )

    # load model and processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.pretrained_lmm_name, dtype=torch.float16, cache_dir=args.cache_dir, device_map="auto")
    processor = AutoProcessor.from_pretrained(args.pretrained_lmm_name)

    # do inference per sample
    final_answers = []
    for idx, batch in enumerate(tqdm(train_dataloader, total=len(train_dataloader))):
        # break code after 200 iterations
        if idx >= 200:
            break

        video_path = batch["video"]
        video = process_videos(zip_path, video_path)

        response = inference(video)

        question = re.search(r"Question:\s*(.+)", (response)).group(1)
        options = dict(re.findall(r"([A-D])\.\s*(.+)", (response)))
        correct_letter = re.search(r"Correct Answer:\s*([A-D])", (response)).group(1)
        correct_answer = options[correct_letter]

        final_answers.append({"video": video_path, "quesion": question, "option": options, "answer": correct_answer})

    json.dump(final_answers, f, indent=2)