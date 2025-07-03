# 如果要限制计算卡编号，请在这里设置，例如只使用 cuda:1-3，如果不用限制，就删除下面这行
# export CUDA_VISIBLE_DEVICES=1,2,3

accelerate launch \
    --num_processes 1 \
    --config_file deepspeed_zero3.yaml \
    train_deepseek-R1-reproduce.py \
    --config deepseek-R1-reproduce.yaml \
    --tf32 False

# 由于我这里使用的是Turing架构的卡，不支持 tf32，因此禁用 tf32，使用 fp32 来计算
