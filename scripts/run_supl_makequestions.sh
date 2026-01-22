export CACHE_DIR="/nfshomes/asarkar6/trinity/model_weights/"
export DATA_DIR="/nfshomes/asarkar6/trinity/music-vqa/"

cd /nfshomes/asarkar6/aditya/audio-video-bench/data_curate/make_supl/

python supl_make_questions.py \
    --pretrained_lmm_name="Qwen/Qwen2.5-VL-7B-Instruct" \
    --data_dir=$DATA_DIR \
    --cache_dir=$CACHE_DIR \
    --batch_size=1 \
    --fps=1.0