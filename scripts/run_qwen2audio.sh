export OUTPUT_DIR="/nfshomes/asarkar6/trinity/music-vqa/"
export CACHE_DIR="/nfshomes/asarkar6/trinity/model_weights/"
export DATA_DIR="/nfshomes/asarkar6/trinity/music-vqa/"

cd /nfshomes/asarkar6/aditya/audio-video-bench/data_curate/

python run_qwen2audio.py \
    --pretrained_lmm_name="Qwen/Qwen2-Audio-7B-Instruct"\
    --data_dir=$DATA_DIR\
    --cache_dir=$CACHE_DIR\
    --batch_size=1\
    --typ="video"