export CUDA_VISIBLE_DEVICES=4

     
# CFG=$1
# filename=$(basename "$CFG")
# dirname=$(dirname "$CFG")
# yamlname="${filename%.*}"
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client tools/vos_inference.py \
    --sam2_cfg //mnt/data/yangyang/code/tracking/sam2-main/sam2_logs/configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml/config_resolved.yaml \
    --sam2_checkpoint sam2_logs/configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune/checkpoints/checkpoint.pt \
    --base_video_dir data/MOSE/valid/JPEGImages \
    --input_mask_dir data/MOSE/valid/Annotations \
    --video_list_file data/MOSE/valid/1.txt \
    --output_mask_dir ./outputs/mose_val_pred_pngs_2
