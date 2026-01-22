# Copyright - UMD

import argparse
import os
from PIL import Image
import re

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import json
from tqdm.auto import tqdm

import pdb

import sys
sys.path.insert(1, "/nfshomes/asarkar6/aditya/audio-video-bench/")
from data_curate.load_music_vqa import music_vqa_dataloader

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

def create_messages(question, options):
    # do the inference
    main_prompt1 = "You are Qwen, created by Alibaba Cloud. You are an intelligent and helpful AI assistant that can do the following task."
    main_prompt2 = "You are given a question and its answer. Your task is to generate three additional options that: \n" 
    main_prompt3 = "1) Are plausible and relevant to the question. \n 2) Match the tone, style, and level of specificity of the existing options. \n 3) Are clearly distinct from the existing options and from each other. \n 4) Do NOT repeat or rephrase the existing options."

    main_prompt4 = "Question: " + question + "\n" + "Answer: " + options
    main_prompt5 = "Generate exactly three new options, labeled 2., 3. and 4. Return only the new options. \n Options:"

    main_prompt = main_prompt1 + "\n" + main_prompt2 + "\n" + main_prompt3 + "\n" + main_prompt4 + "\n" + main_prompt5 

    final_msg = [
        {"role": "system", "content": main_prompt},
    ]
    return final_msg

def inference(question, options):
    # create the conversation
    conversations = create_messages(question, options)

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

    f = open(os.path.join(args.data_dir, "coml_questions_prelim.json"), "w")
    
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
    model = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name_or_path, dtype=torch.float16, cache_dir=args.cache_dir, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

    # do inference per sample
    final_answers = []
    for idx, batch in enumerate(tqdm(train_dataloader, total=len(train_dataloader))):
        if idx >= 200:
            break
        
        # get all the variables
        question = batch["prompt"]
        video = batch["video"]
        answer = batch["answer"]

        response = inference(question[0], answer[0])
        response = {
            int(num): value
            for num, value in re.findall(r"(\d+)\.\s*(.+)", response)
        }
        
        final_answers.append({"video": video, "quesion": question, "option": response, "answer": answer})

    json.dump(final_answers, f, indent=2)