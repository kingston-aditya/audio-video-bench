import numpy as np
import json
import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pdb

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pred_file",
        type=str,
        default="coml_qwen7b_",
        help=(
            "dataset for backgrounds."
        ),
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/nfshomes/asarkar6/trinity/music-vqa/",
        help=(
            "dataset for backgrounds."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/nfshomes/asarkar6/aditya/audio-video-bench/figures/",
        help=(
            "dataset for backgrounds."
        ),
    )

    args = parser.parse_args()
    return args

def process_answer(response):
    return response.strip()

def make_plots(results):
    alphas = [item.upper() for item in types.keys()]

    data = pd.DataFrame({
        "Alphabets": alphas,
        "Accuracy": results 
    })

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=data,
        x="Alphabets",
        y="Accuracy",
        color="salmon"
    )

    # ax.set_title("Values by Category", fontsize=14, weight="bold")
    ax.set_xlabel("Answer choices", fontsize=25)
    ax.set_ylabel("Accuracy", fontsize=25)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)   # thickness
        spine.set_color("black") 

    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"{args.pred_file}_barplot.png"))

def check_answer(pred_file):
    results = []
    for idx, item in enumerate(pred_file.keys()):
        # load the file
        f = open(item)
        js = json.load(f)

        # change it to numpy
        js = np.vectorize(process_answer)(np.asarray(js))
        correct_num = np.sum(js == pred_file[item])/len(js)

        print(correct_num)

        results.append(correct_num)

    make_plots(results)

if __name__ == "__main__":
    args = parse_args()

    types = {"abcd": "A", "pqrs": "P", "bdca": "B", "wqal": "W", "9qj4": "9"}
    
    # get the predicted answers
    pred_file = {os.path.join(args.data_dir, f"{args.pred_file}_{i}.json"):j for i,j in types.items()}

    check_answer(pred_file)
    
    


