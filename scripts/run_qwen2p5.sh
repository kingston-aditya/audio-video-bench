export OUTPUT_DIR="/nfshomes/asarkar6/trinity/music-vqa/"
export CACHE_DIR="/nfshomes/asarkar6/trinity/model_weights/"
export DATA_DIR="/nfshomes/asarkar6/trinity/music-vqa/"

cd /nfshomes/asarkar6/aditya/audio-video-bench/data_curate/

python make_suppl_data.py \
    --pretrained_model_name_or_path="Qwen/Qwen2.5-7B-Instruct"\
    --dataset_name="music"\
    --data_dir=$DATA_DIR\
    --output_dir=$OUTPUT_DIR\
    --cache_dir=$CACHE_DIR\
    --logging_dir=$OUTPUT_DIR\
    --batch_size=1\
    --mixed_precision="fp16"