export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1

# bash tran_dist_debug.sh sam2/configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml

CFG=$1
REL_CFG="${CFG#sam2/}"
echo ${REL_CFG}

python  -m debugpy --listen 0.0.0.0:5679 --wait-for-client training/train.py \
    -c ${REL_CFG}