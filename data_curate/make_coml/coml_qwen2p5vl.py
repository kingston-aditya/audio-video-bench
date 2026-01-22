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
from data_curate.make_coml.load_complement import video_questions_dataloader

import logging
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

    parser.add_argument(
        "--answer_typ",
        type=str,
        default="abcd",
        help="Number of frames to be sampled.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

# create conversation for unimodal and multimodal prompts
def create_conversation(path, question):
    # load all modalities
    prompt = "Select the best answer to the following multiple-choice question based on the video. "
    prompt2 = "Answer with the option\'s letter from the given choices directly and only give the best option. The best answer is: "
    conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [{
                    "type": "video",
                    "video": path,
                    "max_pixels": 360 * 420,
                    "max_frames": 32,
                },
                {"type": "text", "text": prompt + question + "\n" + prompt2},
                ]
            }]
        
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


def inference(video, prompt):
    # create the conversation
    conversations = create_conversation(video, prompt)

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

    f = open(os.path.join(args.data_dir, "coml_qwen2p5vl_answers.json"), "w")
    
    # load dataset
    precomputed_dataset = video_questions_dataloader(args.data_dir, args.answer_typ)
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
        video_path = batch["video"]
        video = process_videos(zip_path, video_path)

        response = inference(video, batch["prompt"][0])

        final_answers.append(response)

    json.dump(final_answers, f, indent=2)