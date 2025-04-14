torchrun \
--nnodes=1 --nproc_per_node=8 --master_port 2328 \
llava/data/extract_token.py \
    --dataset_type "image" \
    --dataset_name "example" \
    --code_path /path_to_saved_tokens \
    --gen_data_path /path_to_meta_json \
    --gen_image_folder "" \
    --gen_resolution 1024
