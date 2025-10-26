export CUDA_VISIBLE_DEVICES=4
export LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH

# bash inference.sh sam2_logs_0803_node4/configs/efficienttam/efficienttam-s-1_sam2.1_hiera_b+_MOSE_finetune.yaml/config_resolved.yaml

full_path=$1

dir_path=$(dirname "$full_path")
file_name=$(basename "$full_path")
export MY_HYDRA_PATH="../$dir_path"

echo "MY_HYDRA_PATH: $MY_HYDRA_PATH"
echo "sam2_cfg     : $file_name"

python tools/vos_inference.py \
    --sam2_cfg $file_name
