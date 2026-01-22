import torch
import argparse
import re
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
import torch.nn.functional as F

import os
from tqdm import tqdm
import sys
sys.path.insert(1, "/nfshomes/asarkar6/aditya/audio-video-bench/")
# load data loaders
from data_curate.make_coml.load_complement import video_questions_dataloader as coml_dataloader
from data_curate.make_supl.load_supplement import video_questions_dataloader as supl_dataloader

# load vocab projection
from analysis.vocab_project import VocabProjectWrapper
from analysis.causal_trace import CausalTraceWrapper

import pdb

tokens_of_interest = ["A", "B", "C", "D", "E"]

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Inference script.")
    parser.add_argument(
        "--pretrained_lmm_name",
        type=str,
        default="Qwen/Qwen2.5-Omni-7B",
        help="Path to LMM.",
    )

    parser.add_argument(
        "--answer_typ",
        type=str,
        default="abcd",
        help="data directory",
    )

    parser.add_argument(
        "--dataset_typ",
        type=str,
        default="coml",
        help="data directory",
    )

    parser.add_argument(
        "--get_logits",
        type=str,
        default="vocabproj",
        help="data directory",
    )

    parser.add_argument(
        "--correctness",
        type=str,
        default='T',
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

def get_required_idxs(answer_typ, correctcy=True):
    # get the base json
    f = open(os.path.join(args.data_dir, "coml_qwen7b_abcd.json"))
    base_json = json.load(f)
    f.close()

    # get the other json
    f = open(os.path.join(args.data_dir, f"coml_qwen7b_{answer_typ}.json"))
    other_json = json.load(f)
    f.close()

    # get correct samples
    reuired_idxs = []
    if len(base_json) == len(other_json):
        if correctcy:
            for idx, item in enumerate(base_json):
                if item == 'A' and other_json[idx] == answer_typ[0].upper():
                    reuired_idxs.append(idx)
        else:
            for idx, item in enumerate(base_json):
                if item == 'A' and other_json[idx] != answer_typ[0].upper():
                    reuired_idxs.append(idx)
        return reuired_idxs
    else:
        raise AssertionError("The lengths must be same!")

# create conversation for unimodal and multimodal prompts
def create_conversation(video, audio, question):
    # load all modalities
    prompt = "Select the best answer to the following multiple-choice question based on the video. "
    prompt2 = "Answer with the option\'s letter or number from the given choices directly and only give the best option. The best answer is: "

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
                {"type": "text", "text": prompt + question[0] +  "\n" + prompt2},
            ],
        }
    ]
        
    return conversation

def inference(video, audio, question, alpha_ids):
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
        if args.get_logits == "vocabproj":
            alpha_logits = vocab_proj_obj.logits_for_alphas(inputs, alpha_ids)
            return alpha_logits
        elif args.get_logits == "causaltrace":
            alpha_logits = causal_trace_obj.logits_for_alphas(inputs, alpha_ids)
            return alpha_logits
        else:
            text_ids = model.generate(**inputs, use_audio_in_video=False)
            text_ids = text_ids[0, inputs["input_ids"].shape[1]:]
            text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            return text[0]


if __name__ == "__main__":
    args = parse_args()
    model_path = args.pretrained_lmm_name

    zip_path = os.path.join(args.data_dir, "music-avqa-synthetic.zip")

    f = open(os.path.join(args.data_dir, f"{args.dataset_typ}_{args.get_logits}_{args.answer_typ}.json"), "w")

    if args.answer_typ == "abcd":
        tokens_of_interest = ["A", "B", "C", "D"]
    elif args.answer_typ == "9qj4":
        tokens_of_interest = ["9", "Q", "J", "4", "A", "B", "C", "D"]
    elif args.answer_typ == "pqrs":
        tokens_of_interest = ["P", "Q", "R", "S", "A", "B", "C", "D"]
    elif args.answer_typ == "wqal":
        tokens_of_interest = ["W", "Q", "A", "L", "B", "C", "D"]
    elif args.answer_typ == "bdca":
        tokens_of_interest = ["B", "D", "C", "A"]
    else:
        raise ValueError("Entered answer type not valid!")
    
    # load dataset
    precomputed_dataset = coml_dataloader(args.data_dir)
    train_dataloader = torch.utils.data.DataLoader(
        precomputed_dataset,
        shuffle=False,
        collate_fn=precomputed_dataset.collate_fn,
        batch_size=1,
        num_workers=1,
    )

    # load model and processor
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(args.pretrained_lmm_name, torch_dtype=torch.float16, cache_dir=args.cache_dir, device_map="auto")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.pretrained_lmm_name)

    alpha_ids = {k:processor.tokenizer.convert_tokens_to_ids(k) for k in tokens_of_interest}

    # load the proj obj
    if args.get_logits == "vocabproj":
        vocab_proj_obj = VocabProjectWrapper(model, processor)
    elif args.get_logits == "causaltrace":
        causal_trace_obj = CausalTraceWrapper(model, processor)
    else:
        pass

    list_idxs = get_required_idxs(args.answer_typ, correctcy=True if args.correctness == 'T' else False)

    pdb.set_trace()

    # do inference per sample
    final_answers = []; final_logits = []
    for idx, batch in enumerate(tqdm(train_dataloader, total=len(train_dataloader))):
        if idx not in list_idxs:
            continue
            
        # get the question, answer
        question = batch["question"]
        video = batch["video"]
        video, audio = process_videos_and_audio(zip_path, video)

        response, probits = inference(video, audio, question, alpha_ids)

        final_logits.append(probits)

    json.dump(final_answers, f, indent=2)