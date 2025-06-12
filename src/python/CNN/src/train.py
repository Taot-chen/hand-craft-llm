from modeling_cnn import SimpleCNN
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 定义超参数
device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = 10
in_channels = 1
learning_rate = 0.001
momentum = 0.9
batch_size = 64
num_epochs = 50

transform = transforms.Compose([
    transforms.ToTensor(),  # 转为张量
    transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
])

def plot_loss(epoch_num, loss):
    plt.figure(figsize=(10, 6))
    plt.plot(
        epoch_num,
        loss, 
        color='red',
        linestyle='-',
        # marker='o',
        linewidth=1,
        label='Training Loss'
    )
    plt.xlim(min(epoch_num), max(epoch_num))
    plt.ylim(0, max(loss)*1.1)
    plt.xlabel('epoch_num', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.title('Loss Curve', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('./loss_curve.png')

def train_cnn():
    print(f"\nTrain on {device}\n")
    # 加载 MNIST 数据集
    train_dataset = datasets.MNIST(root='../data', train=True, transform=transform, download=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化网络
    model = SimpleCNN(in_channels = in_channels, num_classes = num_classes).to(device)

    # 定义损失和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum)
    
    # train
    model.train()

    # 初始化 loss 记录器
    loss_list = []
    epoch_list = []
    min_loss = 1e9
    for epoch in range(num_epochs):
        total_loss = 0
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)

            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, targets)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")
        if total_loss / len(train_loader) < min_loss:
            min_loss = total_loss / len(train_loader)
            torch.save(model.state_dict(), "./ckpt.pt")
        epoch_list.append(epoch + 1)
        loss_list.append(total_loss / len(train_loader))

    plot_loss(epoch_list, loss_list)


def eval_cnn():
    print(f"\nEval on {device}\n")
    test_dataset = datasets.MNIST(root='../data', train=False, transform=transform, download=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
     # 初始化网络
    model = SimpleCNN(in_channels = in_channels, num_classes = num_classes).to(device)
    state_dict = torch.load("./ckpt.pt")
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)            
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # 可视化结果
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    _, predictions = torch.max(outputs, 1)

    fig, axes = plt.subplots(1, 6, figsize=(12, 4))
    for i in range(6):
        axes[i].imshow(images.cpu()[i][0], cmap='gray')
        axes[i].set_title(f"Label: {labels[i]}\nPred: {predictions[i]}")
        axes[i].axis('off')
    plt.savefig('./predict_result.png')
    # plt.show()

if __name__ == "__main__":
    train_cnn()
    eval_cnn()
            