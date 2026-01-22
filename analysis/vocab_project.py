import re

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

"""
Some code borrowed from https://github.com/allenai/understanding_mcqa/blob/main/util/vocab_projection_utils.py
"""

import pdb

class VocabProjectWrapper(nn.Module):
    def __init__(self, model, processor):
        super().__init__()

        self.model = model.eval()
        self.model.config.output_hidden_states = True

        self.tokenizer = processor

        self.layer_names = model.model.layers

        self.num_layers = len(self.layer_names)

    def layer_decode(self, hidden_states):
        logits = []
        for idx, layer in enumerate(hidden_states):
            h = layer[:, -1, :]
        
            if idx == len(hidden_states)-1:
                normed = h
            else:
                normed = self.model.model.norm(h)
            normed=h

            # bring everything to same device.
            normed = normed.to(self.model.lm_head.weight.device)
            l = torch.matmul(self.model.lm_head.weight, normed.T)
            logits.append(l)
        
        return logits
        
    def get_layers(self, inputs):
        with torch.inference_mode():
            outputs = self.model(**inputs, return_dict=True, output_hidden_states=True, use_audio_in_video=False)
        logits = self.layer_decode(outputs.hidden_states)
        return torch.stack(logits)
    
    def logits_for_alphas(self, inputs, alpha_ids, log=True):
        # get the logits
        logits = self.get_layers(inputs)
        if log:
            probits = logits
        else:
            probits = F.softmax(logits, dim=-1)
        topk = []
        for el in range(probits.shape[2]):
            layerwise_topk = {}
            for i, layer in enumerate(probits[:, :, el]):
                layerwise_topk[i] = {key_item:layer[value_item].cpu().numpy() for key_item, value_item in alpha_ids.items()}
            topk.append(layerwise_topk)
        return topk[0]


