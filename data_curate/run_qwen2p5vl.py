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
        "--typ",
        type=str,
        default="gen",
        help="Number of frames to be sampled.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

# create conversation for unimodal and multimodal prompts
def create_conversation(path):
    prompt1 = "Given a video, you need to first understand it and output a question, its answer and THREE additional confusing options in JSON format."
    prompt2 = "Your output must meet the following three requirements: \n 1. Question must NOT be related to the musical instrument. \n 2. It must be grammatically correct. \n. Output must be in JSON format."
    prompt3 = "An example is provided below. \n Example: {\"Question\": \"What is the color of the dress worn by lady?\", \"Answer\": \"red\". , \"Options\" : \"[black, green, yellow]\".}"
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
                {"type": "text", "text": prompt1 + "\n" +  prompt2 + "\n" + prompt3},
            ],
        }
    ]
        
    return conversation

def create_message(path, question):
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
                {"type": "text", "text": question + "Answer:"},
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

def quote_values(match):
    val = match.group(0)
    val = val.strip()
    # If already quoted, number, or starts with [, leave it
    if val.startswith('"') or val.startswith('[') or re.match(r'^[\d.]+$', val):
        return val
    # Otherwise, add quotes
    return f'"{val}"'

def arrange_data(js):
    js = "{" + "".join(js.split("{")[-1].split("}")[0]).strip()+"}"
    s_fixed_keys = re.sub(r'(\b\w+\b)\s*:', r'"\1":', js)
    s_fixed_values = re.sub(r'(?<=: )[^,\]\}]+', quote_values, s_fixed_keys)
    s_fixed_values = re.sub(r'\[([^\]]+)\]', lambda m: '[' + ','.join(f'"{x.strip()}"' for x in m.group(1).split(',')) + ']', s_fixed_values)

    js = js.replace("\n", "")
    js = json.loads(js)

    question = js['Question']
    answer = [js['Answer']]
    options = js['Options'] if isinstance(js['Options'], list) is True else [js["Options"][0]]

    # shuffle the options
    options = options+answer
    np.random.shuffle(options)
    options = [chr(65+idx)+". "+item.strip() for idx, item in enumerate(options)]

    answer_number = chr(65+next((i for i, s in enumerate(options) if answer[0] in s), None))

    # make proper question
    final_part = {"question": question + "\n" + "Options:\n" + "\n".join(options+["E. None of the above."]), "answer": answer_number}

    return final_part

def inference(video, args, question=None):
    # create the conversation
    if args.typ == "gen":
        conversations = create_conversation(video)
    else:
        conversations = create_message(video, question)

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
    f = open(os.path.join(args.data_dir, "video_questions.json"), "w")
    
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

        response = inference(video, args)

        final_answers.append(arrange_data(response))

    json.dump(final_answers, f, indent=2)