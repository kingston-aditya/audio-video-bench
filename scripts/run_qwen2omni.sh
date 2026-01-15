export CACHE_DIR="/nfshomes/asarkar6/trinity/model_weights/"
export DATA_DIR="/nfshomes/asarkar6/trinity/music-vqa/"

cd /nfshomes/asarkar6/aditya/audio-video-bench/pipelines/

python run_qwenomni.py \
    --pretrained_lmm_name="Qwen/Qwen2.5-Omni-7B" \
    --data_dir=$DATA_DIR \
    --cache_dir=$CACHE_DIR \
    --batch_size=1 \
    --fps=1.0 \
    --question_type="normal" \
    --get_logits="yes"