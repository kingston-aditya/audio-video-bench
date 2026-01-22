import torch
import argparse
from PIL import Image
import zipfile
from decord import cpu, VideoReader
import librosa
import tempfile
from moviepy import VideoFileClip
import numpy as np
import json

from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

import os
from tqdm import tqdm
import sys
sys.path.insert(1, "/nfshomes/asarkar6/aditya/audio-video-bench/")
# load data loaders
from data_curate.make_coml.load_complement import video_questions_dataloader as coml_dataloader
from data_curate.make_supl.load_supplement import video_questions_dataloader as supl_dataloader

import pdb

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Inference script.")
    parser.add_argument(
        "--pretrained_lmm_name",
        type=str,
        default="Qwen/Qwen2.5-Omni-7B",
        help="Path to LMM.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="qwen7b",
        help="data directory",
    )

    parser.add_argument(
        "--answer_typ",
        type=str,
        default="explain",
        help="data directory",
    )

    parser.add_argument(
        "--dataset_typ",
        type=str,
        default="coml",
        help="data directory",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/nfshomes/asarkar6/trinity/music-vqa/",
        help="data directory",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/nfshomes/asarkar6/trinity/model_weights/",
        help="cache directory",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Number of frames to be sampled.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

# process video and audio
def process_videos_and_audio(zip_path, video_path, num_frames=32, sampling_rate=16000):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        mini_file_name = "/".join(video_path[0].split("/")[-2:])

        if mini_file_name not in zip_ref.namelist():
            raise FileNotFoundError(f"{mini_file_name} does not exist in zip")
        
        if mini_file_name in zip_ref.namelist():
            with zip_ref.open(mini_file_name) as file:
                # read audio
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name
                    video = VideoFileClip(tmp_path)
                    audio = video.audio
            
            with zip_ref.open(mini_file_name) as file:
                # read video
                vr = VideoReader(file, ctx=cpu(0))

    # process video
    frame_indices = [int(i * len(vr) / 32) for i in range(num_frames)]
    sampled_frames = [Image.fromarray(vr[i].asnumpy()[...,::-1]).resize((224, 224), Image.BILINEAR) for i in frame_indices]

    # process audio
    sr = audio.fps
    audio = audio.to_soundarray()  
    
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)
    audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)
    
    return sampled_frames, audio

# create conversation for unimodal and multimodal prompts
def create_conversation(video, audio, question):
    # load all modalities
    prompt = "You are given a video, its associated audio, and a question. \n Carefully analyze the relevant visual and auditory information needed to answer the question."
    prompt2 = "Output: \n Final Answer: Provide a concise and accurate answer to the question. \n Explanation: Give a short, high-level explanation of the key observations or principles from the video and/or audio that support your answer."
    
    # load all modalities
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video},
                {"type": "audio", "audio": audio},
                {"type": "text", "text": prompt + "\n" + prompt2 + "\n" + question[0]},
            ],
        }
    ]
        
    return conversation


def inference(video, audio, question):
    # create the conversation
    conversations = create_conversation(video, audio, question)

    # do the inference
    text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)

    # compute the inputs
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
    inputs = inputs.to(model.device).to(model.dtype)

    # get the output text
    with torch.no_grad():
        text_ids = model.generate(**inputs, use_audio_in_video=False)
        text_ids = text_ids[0, inputs["input_ids"].shape[1]:]
        text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    torch.cuda.empty_cache()

    return text[0]

if __name__ == "__main__":
    args = parse_args()
    model_path = args.pretrained_lmm_name

    zip_path = os.path.join(args.data_dir, "music-avqa-synthetic.zip")

    f = open(os.path.join(args.data_dir, f"{args.dataset_typ}_{args.model_name}_{args.answer_typ}.json"), "w")
    
    # load dataset
    if args.dataset_typ == "coml":
        precomputed_dataset = coml_dataloader(args.data_dir, options=args.answer_typ)
    else:
        precomputed_dataset = supl_dataloader(args.data_dir, options=args.answer_typ)
    
    train_dataloader = torch.utils.data.DataLoader(
        precomputed_dataset,
        shuffle=False,
        collate_fn=precomputed_dataset.collate_fn,
        batch_size=1,
        num_workers=1,
    )

    # load model and processor
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(args.pretrained_lmm_name, dtype=torch.float16, cache_dir=args.cache_dir, device_map="auto")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.pretrained_lmm_name)

    # do inference per sample
    final_answers = []
    for idx, batch in enumerate(tqdm(train_dataloader, total=len(train_dataloader))):
        # get the question, answer
        question = batch["prompt"]
        video = batch["video"]
        video, audio = process_videos_and_audio(zip_path, video)

        response = inference(video, audio, question)

        final_answers.append(response)

    json.dump(final_answers, f, indent=2)



    








