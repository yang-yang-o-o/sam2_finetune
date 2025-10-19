# export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH

# bash train_dist.sh sam2/configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml

CFG=$1
REL_CFG="${CFG#sam2/}"
echo ${REL_CFG}

python training/train.py \
    -c ${REL_CFG}