# bash tensorboard.sh /mnt/data/yangyang/code/tracking/sam2-main/sam2_logs/configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml/tensorboard

LOGDIR=$1

tensorboard \
    --logdir ${LOGDIR} \
    --port 6006
