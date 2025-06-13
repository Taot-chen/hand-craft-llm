import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from modeling_rnn_torch_nn_rnn import SimpleRNN
from modeling_rnn_craft import CustomRNN


def plot_loss(loss, pic_path = './loss_curve.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(
        loss, 
        color='red',
        linestyle='-',
        # marker='o',
        linewidth=1,
        label='Training Loss'
    )
    plt.ylim(0, max(loss)*1.1)
    plt.xlabel('epoch_num', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.title('Loss Curve', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(pic_path)

# --------------------------------------------------------------------------------
# 数据集：字符序列预测（Hello -> Elloh）
char_set = list("hello")
char_to_idx = {c: i for i, c in enumerate(char_set)}
idx_to_char = {i: c for i, c in enumerate(char_set)}

input_str = "hello"
target_str = "elloh"
input_data = [char_to_idx[c] for c in input_str]
target_data = [char_to_idx[c] for c in target_str]

# 转换为独热编码
input_one_hot = np.eye(len(char_set))[input_data]

# 转换为 PyTorch Tensor
inputs = torch.tensor(input_one_hot, dtype=torch.float32)
targets = torch.tensor(target_data, dtype=torch.long)



# --------------------------------------------------------------------------------
# 超参
device = "cuda" if torch.cuda.is_available() else "cpu"
input_size = len(char_set)
hidden_size = 8
output_size = len(char_set)
num_epochs = 1000
learning_rate = 0.01



# --------------------------------------------------------------------------------
# train and eval
def train_model(model, model_type = "craft", inputs = None, targets = None):
    print(f"\nTrain on {device}\n")
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"⚠️ 参数未启用梯度: {name}")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    hidden_state = None
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # 前向传播
        if model_type == "craft":
            outputs = []
            for index in range(inputs.shape[0]):
                output, hidden_state = model(inputs[index], hidden_state)
                hidden_state = hidden_state.detach()    # 防止梯度爆炸
                outputs.append(output)
            outputs = torch.stack(outputs, dim=0).squeeze(1)
        else:
            outputs = None
            for index in range(inputs.shape[0]):
                output, hidden_state = model(inputs.unsqueeze(0), hidden_state)
                hidden_state = hidden_state.detach()  # 防止梯度爆炸
                outputs = torch.cat((outputs, output), dim=0) if outputs is not None else output

        # 计算损失
        loss = criterion(outputs, targets)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    
    # 测试
    with torch.no_grad():
        test_hidden = None
        if model_type == "craft":
            predicts = []
            for index in range(inputs.shape[0]):
                test_output, _ = model(inputs[index].unsqueeze(0), test_hidden)
                if device != "cpu":
                    predicted = torch.argmax(test_output, dim=1).item()
                else:
                    predicted = torch.argmax(test_output, dim=1)
                predicts.append(predicted)
                
            print("Input sequence: ", ''.join([idx_to_char[i] for i in input_data]))
            print("Predicted sequence: ", ''.join([idx_to_char[i] for i in predicts]))
            print("Target sequence: ", target_str)
        else:
            test_output, _ = model(inputs.unsqueeze(0), hidden_state)
            if device != "cpu":
                predicts = torch.argmax(test_output, dim=1).cpu().squeeze().numpy()
            else:
                predicts = torch.argmax(test_output, dim=1).squeeze().numpy()

    plot_loss(losses, f"loss_curve_{model_type}.png")

if __name__ == "__main__":
    model_craft = CustomRNN(input_size, hidden_size, output_size).to(device)
    model_torch_rnn = SimpleRNN(input_size, hidden_size, output_size).to(device)
    train_model(model_craft, "craft", inputs = inputs.to(device), targets = targets.to(device))
    train_model(model_torch_rnn, "torch_rnn", inputs = inputs.to(device), targets = targets.to(device))
