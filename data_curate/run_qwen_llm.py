# Copyright - UMD

import argparse
import math
import os
from pathlib import Path
from PIL import Image
import re

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

import json
from tqdm.auto import tqdm

import pdb

import sys
sys.path.insert(1, "/nfshomes/asarkar6/aditya/audio-video-bench/")
from data_curate.load_summaries import summary_dataloader

# class ForkedPdb(pdb_original.Pdb):
#     """A Pdb subclass that may be used
#     from a forked multiprocessing child
#     """
#     def interaction(self, *args, **kwargs):
#         _stdin = sys.stdin
#         try:
#             sys.stdin = open('/dev/stdin')
#             pdb_original.Pdb.interaction(self, *args, **kwargs)
#         finally:
#             sys.stdin = _stdin

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/nfshomes/asarkar6/aditya/PRISM/backgrounds/",
        help=(
            "dataset for backgrounds."
        ),
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/nfshomes/asarkar6/trinity/",
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )

    args = parser.parse_args()
    return args

def create_messages(video_summary, audio_summary, typ):
    # do the inference
    main_prompt1 = "You are Qwen, created by Alibaba Cloud. You are an intelligent and helpful AI assistant that can do the following task."
    main_prompt2 = "Given a video summary, you need to first understand it and output a question, its answer and THREE additional confusing options in JSON format." 
    main_prompt2a = "Ask something about the visual part like color of some object or some visual thing."

    if typ == "video":
        main_prompt3 = "Your output must meet the following two requirements: \n 1. Question must be from something that can be answered ONLY from video. \n 2. It must be grammatically correct. \n. Output must be in JSON format."
        main_prompt4 = "Example 1, Video summary: A lady is playing a violin. \n {\"Question\": \"What is the color of the dress worn by lady?\", \"Answer\": \"red\". , \"Options\" : \"[black, green, yellow]\".} \n Explanation: both video and audio has violin. So the question is related to lady's dress. Also there are three options."
        main_prompt5 = "Example 2, Video summary: A dog is barking at the cat near a house. \n {\"Question\": \"What is the color of the house?\", \"Answer\": \"white\"., \"Options\" : \"[black, blue, grey]\".} \n Explanation: both video and audio has a barking dog. So the question is related to house. Also there are three options."
    else:
        pass

    main_prompt6 = "Just give me the question, answer and the THREE options in JSON format, and do NOT give any explanations."
    main_prompt = main_prompt1 + "\n" + main_prompt2 + main_prompt2a + "\n" + main_prompt3 + "\n" + main_prompt4 + "\n" + main_prompt5 + "\n" + main_prompt6

    final_msg = [
        {"role": "system", "content": main_prompt},
        {"role": "user", "content": "Video summary:" + video_summary + "\n" + "Question:"},
    ]
    return final_msg

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

def inference(video_summary, audio_summary):
    # create the conversation
    conversations = create_messages(video_summary, audio_summary, typ="video")

    text = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # get the output text
    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, max_new_tokens=512)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    torch.cuda.empty_cache()
    return response[0]

if __name__ == "__main__":
    args = parse_args()

    f = open(os.path.join(args.data_dir, "final_questions.json"), "w")
    
    # load dataset
    precomputed_dataset = summary_dataloader(args.data_dir)
    train_dataloader = torch.utils.data.DataLoader(
        precomputed_dataset,
        shuffle=False,
        collate_fn=precomputed_dataset.collate_fn,
        batch_size=args.batch_size,
        num_workers=1,
    )

    # load model and processor
    model = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name_or_path, dtype=torch.float16, cache_dir=args.cache_dir, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

    # do inference per sample
    final_answers = []
    for idx, batch in enumerate(tqdm(train_dataloader, total=len(train_dataloader))):
        video_summary = batch["video"]
        audio_summary = batch["audio"]

        # try:
        response = inference(video_summary[0], audio_summary[0])
        response = arrange_data(response)
        # except:
        #     raise ValueError(f"Error at batch {idx}.")
        
        final_answers.append(response)

    json.dump(final_answers, f, indent=2)