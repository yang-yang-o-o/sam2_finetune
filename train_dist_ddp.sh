export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH

export PYTHONUNBUFFERED=1
# export NCCL_IB_HCA=mlx5_0:7
export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0

# bash tran_dist_ddp.sh 0 sam2/configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml

NODE_RANK=$1
CFG=$2
REL_CFG="${CFG#sam2/}"
echo ${REL_CFG}

export NODE_RANK=$1

torchrun \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=$NODE_RANK \
    --master_addr="10.112.1.88" \
    --master_port=12345 \
    training/train.py  --c ${REL_CFG} #&
    # | tee ./results/${yamlname}/train_output_worker_$NODE_RANK.log &
