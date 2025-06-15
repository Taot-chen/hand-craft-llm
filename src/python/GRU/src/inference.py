import torch
from modeling_gru import GRUModel
from train import preprocess_data, label_dict


def predict(tokens, model, device, char_to_idx, max_length = 256):
    model = model.to(device)

    inputs = []
    for sentence in tokens:
        part = [char_to_idx[char] if char in char_to_idx else 0 for char in sentence]
        inputs.append(part)

    # 处理文本
    new_inputs = []
    for token in inputs:
        # 截断或填充标签以匹配最大序列长度
        new_input = token[:max_length] + [input_size - 1] * (max_length - len(token[:max_length]))

        # 将标签列表转换为LongTensor 
        new_input = torch.LongTensor (new_input)
        new_inputs.append(new_input)

    preds = []
    with torch.no_grad():
        for batch, inputs in enumerate(new_inputs):
            inputs = inputs.to(device)
            size = inputs.size(0)
            outs = model(inputs)
            outs = outs.argmax(dim=-1)     
            preds.append(outs)

    preds_txt = []
    for index, pred in enumerate(preds):
        pred = pred[:len(tokens[index])]
        idx_to_label = {v: k for k, v in label_dict.items()}
        label_strings = [idx_to_label[idx.item()] for idx in pred]
        preds_txt.append(label_strings)
    return pred, preds_txt

def infer_model(tokens, input_size, num_class, hidden_size, device, char_to_idx):
    model = GRUModel(input_size, num_class, hidden_size).to(device)
    model_state_dict = torch.load("../model_best.pth")
    model.load_state_dict(model_state_dict)

    preds, preds_txt = predict(tokens, model, device, char_to_idx)
    for idx, token in enumerate(tokens):
        for num in range(len(token)):
            print(token[num], preds_txt[idx][num])
        print('\n\n')

if __name__ == "__main__":
    tokens = [
        '在2024年的春天，张伟、李娜和王强三位科学家在中国科学院的资助下，前往云南的西双版纳热带植物园进行生物多样性研究。他们与当地的傣族村民合作，共同探索了这片土地上丰富的植物资源，希望能找到对抗气候变化的新方法。' ,
        '正当朱镕基当选政府总理后第一次在中外记者招待会上，回答外国记者的提问：中国农村是否实行民主选举制度的时候，一位中国电视编导正携带着她的反映中国农村民主选举村委会领导的电视纪录片《村民的选择》（北京电视台摄制，仝丽编导），出现在法国的真实电影节上。',
        '中共中央致中国致公党十一大的贺词各位代表、各位同志：在中国致公党第十一次全国代表大会隆重召开之际，中国共产党中央委员会谨向大会表示热烈的祝贺，向致公党的同志们致以亲切的问候！'
    ]
    train_data = "../data/msra_train_bio.txt"
    test_data = "../data/msra_test_bio.txt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    char_to_idx, token_counts = preprocess_data(train_data)
    input_size = len(token_counts) + 2
    num_class = len(label_dict)
    hidden_size = 768
    infer_model(tokens, input_size, num_class, hidden_size, device, char_to_idx)
