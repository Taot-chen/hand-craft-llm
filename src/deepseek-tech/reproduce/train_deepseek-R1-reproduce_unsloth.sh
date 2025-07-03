# 如果你要限制计算卡编号，请在这里设置
export CUDA_VISIBLE_DEVICES=0

python3 train_deepseek-R1-reproduce_unsloth.py --config deepseek-R1-reproduce_unsloth.yaml
