import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import torch

from moviepy import VideoFileClip
import numpy as np
import tempfile

import os
import sys
sys.path.insert(1, "/nfshomes/asarkar6/aditya/audio-video-bench/")
from data_curate.load_complement import video_questions_dataloader

import json
import argparse
import zipfile
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
        default=None,
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
def create_conversation(audio):
    # load audio modalities
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": "Where is the sound coming from?"},
            ],
        }
    ]
        
    return conversation

def create_message(audio, question):
    prompt = "Answer the question by choosing the best option. Do NOT output any explanation."
    # load audio modalities
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": prompt + "\n" + question + "\n" + "Answer:"},
            ],
        }
    ]
    return conversation

def get_audio_from_video(zip_path, video_path, sampling_rate=16000):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        mini_file_name = "/".join(video_path[0].split("/")[-2:])

        if mini_file_name not in zip_ref.namelist():
            raise FileNotFoundError(f"{mini_file_name} does not exist in zip")

        if mini_file_name in zip_ref.namelist():
            with zip_ref.open(mini_file_name) as file:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name
                    video = VideoFileClip(tmp_path)
                    audio = video.audio

        # librosa type
        sr = audio.fps
        audio = audio.to_soundarray()  
        
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        audio = audio.astype(np.float32)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)
    
    return audio


def inference(audio, args, question=None):
    # create the conversation
    if args.typ == "gen":
        conversations = create_conversation(audio)
    else:
        conversations = create_message(audio, question)

    # process the text
    text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)

    # compute the inputs
    inputs = processor(text=text, audio=audio, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.no_grad():
    # get the output text
        generate_ids = model.generate(**inputs, max_length=2048)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return response

if __name__ == "__main__":
    args = parse_args()
    model_path = args.pretrained_lmm_name

    zip_path = os.path.join(args.data_dir, "music-avqa-synthetic.zip")
    f = open(os.path.join(args.data_dir, "audio_results.json"), "w")
    
    # load dataset
    precomputed_dataset = video_questions_dataloader(args.data_dir)
    train_dataloader = torch.utils.data.DataLoader(
        precomputed_dataset,
        shuffle=False,
        collate_fn=precomputed_dataset.collate_fn,
        batch_size=args.batch_size,
        num_workers=1,
    )

    # load model and processor
    model = Qwen2AudioForConditionalGeneration.from_pretrained(args.pretrained_lmm_name, dtype=torch.float16, cache_dir=args.cache_dir, device_map="auto")
    processor = AutoProcessor.from_pretrained(args.pretrained_lmm_name)

    # do inference per sample
    final_answers = []
    for idx, batch in enumerate(tqdm(train_dataloader, total=len(train_dataloader))):
        if idx >= 200:
            break
        
        video_path = batch["video"]
        audio = get_audio_from_video(zip_path, video_path, processor.feature_extractor.sampling_rate)

        response = inference(audio, args, batch["question"][0])
        final_answers.append(response)

    json.dump(final_answers, f, indent=2)



    








