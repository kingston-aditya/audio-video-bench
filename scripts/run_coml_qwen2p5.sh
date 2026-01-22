export CACHE_DIR="/nfshomes/asarkar6/trinity/model_weights/"
export DATA_DIR="/nfshomes/asarkar6/trinity/music-vqa/"

cd /nfshomes/asarkar6/aditya/audio-video-bench/data_curate/make_coml/

python coml_qwen2p5.py \
    --pretrained_model_name_or_path="Qwen/Qwen2.5-7B-Instruct"\
    --data_dir=$DATA_DIR\
    --cache_dir=$CACHE_DIR\
    --batch_size=1