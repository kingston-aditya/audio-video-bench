export DATA_DIR="/nfshomes/asarkar6/trinity/music-vqa/"
export OUTPUT_DIR="/nfshomes/asarkar6/aditya/audio-video-bench/figures/"

cd /nfshomes/asarkar6/aditya/audio-video-bench/analysis/

python check_answer.py \
    --pred_file="coml_qwen7b" \
    --data_dir=$DATA_DIR \
    --output_dir=$OUTPUT_DIR \