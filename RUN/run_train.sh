PER_GPU_BATCH_SIZE=16
PER_GPU_EVAL_BATCH_SIZE=8
GC_STEP=2
GPU_NUM=1
lr_iter=(1 2 5) #  (1 0.5 2)
#LR_RATE=1
epoch_iter=5
#EPOCH_N=6
SEED=42 # seed_iter=(42 100 512 1024 2019)
MODE='train'
TASK_NAME='baseline'
M_NAME='blender_small'
MODEL_PATH=/data/pretrained_models/blender_small
CACHE_FILE='cached'
DATA_DIR=/data//fourth_next/phy_diag/data
OUTPUT_BASE=/data//fourth_next/phy_diag/blender_strategy
for LR_RATE in ${lr_iter[@]}; do
    for EPOCH_N in ${epoch_iter[@]}; do
        EXP_PRE="seed${SEED}"
        OUTPUT_DIR=${OUTPUT_BASE}/${TASK_NAME}_${M_NAME}_gn${GPU_NUM}_bp${PER_GPU_BATCH_SIZE}_gc${GC_STEP}_lr${LR_RATE}_e${EPOCH_N}_${EXP_PRE}
        mkdir -p ${OUTPUT_DIR}
        CUDA_VISIBLE_DEVICES=0 \
        python BlenderEmotionalSupport.py \
        --model_type ${TASK_NAME} \
        --cache_dir ${DATA_DIR}/${CACHE_FILE} \
        --model_name_or_path ${MODEL_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --mode ${MODE} \
        --do_train \
        --do_eval \
        --evaluate_during_training \
        --overwrite_output_dir \
        --strategy \
        --per_gpu_train_batch_size ${PER_GPU_BATCH_SIZE} \
        --per_gpu_eval_batch_size ${PER_GPU_EVAL_BATCH_SIZE} \
        --num_train_epochs ${EPOCH_N} \
        --max_steps -1 \
        --learning_rate ${LR_RATE}e-5 \
        --gradient_accumulation_steps ${GC_STEP} \
        --weight_decay 0.0 \
        --adam_epsilon 1e-8 \
        --max_grad_norm 1.0 \
        --block_size 512 \
        --warmup_steps 120 \
        --logging_steps 30 \
        --save_steps 120 \
        --seed ${SEED}
    done
done