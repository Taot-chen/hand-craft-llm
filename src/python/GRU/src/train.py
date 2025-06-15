from torch.utils.data import Dataset
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import time
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from collections import Counter
from sklearn.metrics import f1_score
from modeling_gru import GRUModel


# 设置matplotlib配置参数
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
# plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 人名(PER)、地名(LOC)和机构名(ORG)
# label_dict = {
#     'O': 0,
#     'B-PER': 1,
#     'I-PER': 2,
#     'B-LOC': 3,
#     'I-LOC': 4,
#     'B-ORG': 5,
#     'I-ORG': 6,
#     'padding': 7
# }
label_dict = {
    'O': 0,
    'PER': 1,
    'LOC': 2,
    'ORG': 3,
    'padding': 4
}


def preprocess_data(file):
    # 读取文本文件
    print("file: ", file)
    with open(file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 用于存储最终的实体和标签
        texts = []

    text = []
    for line in lines:
        if line == '\n':
            texts.append(text)

            text = []
            label = []
        else:
            parts = line.split('\t')
            text.append(parts[0])

    # print("len(texts): ", len(texts))
    # print("texts:\n", texts)

    # 统计词频
    # 展平二维列表
    all_tokens = [token for sublist in texts for token in sublist]

    # 统计词频
    token_counts = Counter(all_tokens)

    # print("len(token_counts): ", len(token_counts))

    # 创建一个字典，将每个字符映射到一个唯一的整数
    char_to_idx = {char: i + 1 for i, char in enumerate(set(''.join(all_tokens)).union({'<PAD>'}))}

    return char_to_idx, token_counts

def plot_text(dataset, figpath):
    # 收集所有文本长度
    lengths = [len(text) for text, label in dataset]

    # 计算最大长度
    max_length = max(lengths)
    # print("最大文本长度:", max_length)

    # 绘制直方图
    plt.hist(lengths, bins=100, edgecolor='black')
    plt.title('文本长度分布')
    plt.xlabel('文本长度')
    plt.ylabel('频数')
    plt.grid(True)
    # plt.show()
    plt.savefig(figpath)


def collate_fn(data, max_length = 256):
    tokens = [i[0] for i in data]  # 提取所有文本
    labels = [i[1] for i in data]  # 提取所有标签
    
    # 处理文本
    new_tokens = []
    for token in tokens:
        # 截断或填充标签以匹配最大序列长度
        new_token = token[:max_length] + [input_size - 1] * (max_length - len(token[:max_length]))
        new_tokens.append(new_token)

    # 将标签列表转换为LongTensor
    tokens_tensor = torch.LongTensor(new_tokens)

    # 处理标签
    new_labels = []
    for label in labels:
        # 截断或填充标签以匹配最大序列长度
        new_label = label[:max_length] + [num_class - 1] * (max_length - len(label[:max_length]))
        new_labels.append(new_label)

    # 将标签列表转换为LongTensor
    labels_tensor = torch.LongTensor(new_labels)

    return tokens_tensor, labels_tensor


def get_correct_and_total_count(labels, outs):
    outs = outs.argmax(dim=2)                   # 获取概率最高的类别作为预测结果
    correct = (outs == labels).sum().item()     # 计算总的正确预测数量
    total = labels.size(0) * labels.size(1)     # 计算总的标签数量

    # 计算除了补充元素以外的正确率
    select = labels != num_class - 1                  # 选择非补充的标签

    labels_non_supplementary = labels[select]                     # 选择非补充的真实标签
    outs_non_supplementary = outs[select]                         # 选择非补充的预测结果

    # 计算F1分数
    # F1分数是衡量分类模型精确度的一种指标，它是精确率和召回率的调和平均数，用于在模型的精确性和完整性之间取得平衡。
    f1 = f1_score(labels_non_supplementary.cpu().numpy(), outs_non_supplementary.cpu().numpy(), average='macro')

    # 由于labels和outs在过滤后可能不再是二维的，需要调整它们的形状
    labels_non_supplementary = labels_non_supplementary.view(-1)  # 展平为一维
    outs_non_supplementary = outs_non_supplementary.view(-1)  # 展平为一维
    correct_content = (outs_non_supplementary == labels_non_supplementary).sum().item()      # 计算非补充部分的正确预测数量
    total_content = len(labels_non_supplementary)                 # 计算非补充部分的标签数量

    return correct, total, correct_content, total_content, f1


class CustomDataset(Dataset):
    def __init__(self, file):
        super(CustomDataset, self).__init__()
        char_to_idx, token_counts = preprocess_data(file)

        # 读取文本文件
        with open(file, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # 用于存储最终的实体和标签
        texts = []
        labels = []

        text = []
        label = []
        for line in lines:
            if line == '\n':
                # 将每个字符转换为整数
                encoded_texts = [char_to_idx[char] if char in char_to_idx else 0 for char in text]
                texts.append(encoded_texts)
                try:
                    labels.append([label_dict[num] if num != '' else 0 for num in label])
                except:
                    pass
                text = []
                label = []
            else:
                parts = line.split('\t')
                text.append(parts[0])
                label.append(parts[-1].strip() if parts[-1].strip() == 'O' else parts[-1].strip()[2:])

        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return self.texts[item], self.labels[item]


def train_model(dataloader, model, criterion, optimizer, scheduler, device):
    size = len(dataloader)
    avg_loss = 0
    correct, total, correct_content, total_content = 0, 0, 0, 0

    model.train()
    # for batch, (inputs, labels) in enumerate(tqdm(dataloader, desc="Processing")):
    for batch, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向
        out = model(inputs)
        loss = criterion(out.view(-1, num_class), labels.view(-1))
        avg_loss += loss

        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        counts = get_correct_and_total_count(labels, out)

        correct += counts[0]
        total += counts[1]
        correct_content += counts[2]
        total_content += counts[3]

    # 更新学习率
    scheduler.step()

    # 一个epoch完了后返回平均 loss
    avg_loss /= size
    avg_loss = avg_loss.detach().cpu().numpy()
    accuracy = correct / total
    accuracy_content = correct_content / total_content

    print(f"train: correct = {100 * accuracy:.3f}%, Accuracy: {(100 * accuracy_content):>0.3f}%, Avg loss: {avg_loss:>8f}, f1_score: {counts[4]:>5f}")
    return accuracy_content, avg_loss


def validate(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)

    # 将模型转为验证模式
    model.eval()

    # 初始化test_loss 和 correct， 用来统计每次的误差
    test_loss = 0
    correct, total, correct_content, total_content = 0, 0, 0, 0

    with torch.no_grad():
        for batch, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outs = model(inputs)
            loss = loss_fn(outs.view(-1, num_class) , labels.view(-1))
            test_loss += loss
            counts = get_correct_and_total_count(labels, outs)

            correct += counts[0]
            total += counts[1]
            correct_content += counts[2]
            total_content += counts[3]

    test_loss /= size
    test_loss = test_loss.detach().cpu().numpy()
    accuracy = correct / total
    accuracy_content = correct_content / total_content
    print(f"valid: correct = {100 * accuracy:.3f}%, Accuracy: {(100 * accuracy_content):>0.3f}%, Avg loss: {test_loss:>8f}, f1_score: {counts[4]:>5f}")
    return accuracy_content, test_loss


def launch_train(device, train_data, test_data, batch_size, input_size, num_class, hidden_size, middle_hidden_size):
    epochs = 5
    loss_ = 100
    avg_loss = 0

    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []

    train_data = CustomDataset(train_data)
    valid_data = CustomDataset(test_data)

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    valid_dataloader = DataLoader(
        dataset=valid_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True
    )

    # 输出使用设备
    print(f"Using {device} device")

    model = GRUModel(input_size, num_class, hidden_size, middle_hidden_size, device)
    model = model.to(device)

    # 定义标签损失权重
    class_weights = torch.tensor([1, 10, 10, 10, 0.01]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # 定义优化器
    lr = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 定义学习率策略
    # scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-10)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.95)

    for t in range(epochs):
        print(f"\n----------------------Training Epoch {t} ----------------------------")

        train_accuracy, avg_loss = train_model(train_dataloader, model, loss_fn, optimizer, scheduler, device)
        val_accuracy, val_loss = validate(valid_dataloader, model, loss_fn, device)

        train_accs.append(train_accuracy) 
        train_losses.append(avg_loss)
        val_accs.append(val_accuracy)
        val_losses.append(val_loss)

        if avg_loss < loss_:
            loss_ = avg_loss
            torch.save(model.state_dict(), "../model_best.pth")

        # torch.save(model.state_dict(), "model_last.pth")

    # 绘制训练损失和验证损失曲线
    plt.figure(figsize=(4, 3))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制训练损失和验证正确率曲线
    plt.figure(figsize=(4, 3))
    plt.plot(train_accs, label='Train Accuracy', color='blue')
    plt.plot(val_accs, label='Validation Accuracy', color='red')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.legend()

    # 显示图形
    # plt.show()
    plt.savefig("../lose_curve.png")


if __name__ == "__main__":
    train_data = "../data/msra_train_bio.txt"
    test_data = "../data/msra_test_bio.txt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128
    seq_length = 256
    _, token_counts = preprocess_data(train_data)
    input_size = len(token_counts) + 2
    num_class = len(label_dict)
    hidden_size = input_size
    middle_hidden_size = 256
    launch_train(device, train_data, test_data, batch_size, input_size, num_class, hidden_size, middle_hidden_size)
