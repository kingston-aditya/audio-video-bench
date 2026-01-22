import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import os

"""
Some code borrowed from https://github.com/allenai/understanding_mcqa/blob/main/util/vocab_projection_utils.py
"""

import pdb

class CausalTraceWrapper(nn.Module):
    def __init__(self, model, processor):
        super().__init__()

        self.model = model.eval()
        self.model.config.output_hidden_states = True

        self.tokenizer = processor

        self.layer_names = model.model.layers

        self.num_layers = len(self.layer_names)

    def save_hidden_state(self, inputs, data_dir):
        # make directory
        os.makedirs(os.path.join(data_dir, "hidden_states"))

        with torch.inference_mode():
            outputs = self.model(**inputs, return_dict=True, output_hidden_states=True, use_audio_in_video=False)
            hidden_states = outputs.hidden_states

            temp = []
            for idx, layer in enumerate(hidden_states):
                h = layer[:, -1, :]
            
                if idx == len(hidden_states)-1:
                    normed = h
                else:
                    normed = self.model.model.norm(h)
                
                temp.append(normed)
            temp = torch.stack(temp)

            torch.save(temp, os.path.join(os.path.join(data_dir, "hidden_states"), f'hidden_state_{idx}.pt'))
                
    def load_hidden_state(self, data_dir, idx):
        # make directory
        path = os.path.join(data_dir, "hidden_states")

        # get the loaded tensor
        loaded_tensor = torch.load(os.path.join(path, f'hidden_state_{idx}.pt'))
        return loaded_tensor
    
    def hook(self, hidden_state):
        def replace_hidden_state(module, input, output):
            temp = (hidden_state,)
            return temp
        return replace_hidden_state

    def logits_for_alphas(self, inputs, alpha_ids, hidden_state=None):
        new_embedding = torch.randn(1, 3907, 3584, device=self.model.device, dtype=self.model.dtype)
        temp = {}
        for idx, layer in enumerate(self.layer_names):
            new_hook = layer.register_forward_hook(self.hook(new_embedding))
            with torch.no_grad():
                outputs = self.model(**inputs, return_dict=True, output_hidden_states=True, use_audio_in_video=False)
                logits = outputs.logits
            pdb.set_trace()
            temp[idx] = {key_item:logits[:,-1,:][0][value_item].cpu().numpy() for key_item, value_item in alpha_ids.items()}
        return temp

    
    


    


