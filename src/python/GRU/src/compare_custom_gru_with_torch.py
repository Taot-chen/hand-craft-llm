import torch.nn as nn
import torch
from modeling_gru import CustomGRU

def compare_gru(input_size, hidden_size, seq_length, batch_size, device = None):
    # 将nn.GRU中的4个随机初始化的可学习参数进行保存，并用来替换CustomGRU中CustomGRU_layer随机初始化的可学习参数，并通过torch.allclose判断输出是否相等，若相等则证明MyGRU的实现与官方的nn.GRU是一致的
    # 初始化nn.GRU
    torch_gru = nn.GRU(input_size = input_size, hidden_size = hidden_size).to(device)
    weight_ih_l0 = torch_gru.weight_ih_l0.T
    weight_hh_l0 = torch_gru.weight_hh_l0.T
    bias_ih_l0 = torch_gru.bias_ih_l0
    bias_hh_l0 = torch_gru.bias_hh_l0

    # 初始化CustomGRU
    custom_gru = CustomGRU(input_size = input_size, hidden_size = hidden_size, device = device).to(device)

    # 替换CustomGRU中的参数
    # 更新门的输入权重
    custom_gru.gru_W_xr = nn.Parameter(weight_ih_l0[:, :custom_gru.gru.W_xr.size(1)])
    # 更新门的隐藏权重
    custom_gru.gru.W_hr = nn.Parameter(weight_hh_l0[:, :custom_gru.gru.W_hr.size(1)])
    # 重置门的输入权重
    custom_gru.gru.W_xz = nn.Parameter(weight_ih_l0[:, custom_gru.gru.W_xr.size(1):custom_gru.gru.W_xr.size(1) + custom_gru.gru.W_xz.size(1)])
    # 重置门的隐藏权重
    custom_gru.gru.W_hz = nn.Parameter(weight_hh_l0[:, custom_gru.gru.W_hr.size(1):custom_gru.gru.W_hr.size(1) + custom_gru.gru.W_hz.size(1)])
    # 候选隐藏状态的输入权重
    custom_gru.gru.W_xh = nn.Parameter(weight_ih_l0[:, custom_gru.gru.W_xr.size(1) + custom_gru.gru.W_xz.size(1):])
    # 候选隐藏状态的隐藏权重
    custom_gru.gru.W_hh = nn.Parameter(weight_hh_l0[:, custom_gru.gru.W_hr.size(1) + custom_gru.gru.W_hz.size(1):])

    # 更新门的偏置
    custom_gru.gru.hb_r = nn.Parameter(bias_hh_l0[:custom_gru.gru.hb_r.size(0)])
    # 重置门的偏置
    custom_gru.gru.hb_z = nn.Parameter(bias_hh_l0[custom_gru.gru.hb_r.size(0):custom_gru.gru.hb_z.size(0) + custom_gru.gru.hb_r.size(0)])
    # 候选隐藏状态的偏置
    custom_gru.gru.hb_h = nn.Parameter(bias_hh_l0[custom_gru.gru.hb_z.size(0) + custom_gru.gru.hb_r.size(0):])

    custom_gru.gru.xb_r = nn.Parameter(bias_ih_l0[:custom_gru.gru.xb_r.size(0)])
    custom_gru.gru.xb_z = nn.Parameter(bias_ih_l0[custom_gru.gru.xb_r.size(0):custom_gru.gru.xb_z.size(0) + custom_gru.gru.xb_r.size(0)])
    custom_gru.gru.xb_h = nn.Parameter(bias_ih_l0[custom_gru.gru.xb_z.size(0) + custom_gru.gru.xb_r.size(0):])

    # 初始化输入数据
    x = torch.rand(seq_length, batch_size, input_size).to(device)

    # 获取CustomGRU和nn.GRU的输出
    output1, h1 = custom_gru(x)
    output2, h2 = torch_gru(x)

    # 使用torch.allclose比较输出是否相等
    print("output1 == output2 ? {}".format(torch.allclose(output1.to('cpu'), output2.to('cpu'), atol=1e-2)))
    print("h1 == h2 ? {}".format(torch.allclose(h1.to('cpu'), h2.to('cpu'), atol=1e-2)))

if __name__ == "__main__":
    compare_gru(
        input_size = 32,
        hidden_size = 64,
        seq_length = 32,
        batch_size = 16,
        device = "cuda" if torch.cuda.is_available() else "cpu"
    )
