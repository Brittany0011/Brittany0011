import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
# 用于加载bert模型的分词器
from transformers import AutoTokenizer
# 用于加载bert模型
from transformers import BertModel
from pathlib import Path
import torchvision
batch_size = 16
# 文本的最大长度
text_max_length = 128
# 总训练的epochs数，我只是随便定义了个数
epochs = 2
# 学习率
lr = 3e-5
# 取多少训练集的数据作为验证集
validation_ratio = 0.4
# 启用设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 每多少步，打印一次loss
log_per_step = 50

# 数据集所在位置，如果数据集位置不存在就创建一个空的
dataset_dir = Path("./基于论文摘要的文本分类与关键词抽取挑战赛公开数据")
os.makedirs(dataset_dir) if not     os.path.exists(dataset_dir) else ''

# 模型存储路径
model_dir = Path("./model/bert_checkpoints")
# 如果模型目录不存在，则创建一个
os.makedirs(model_dir) if not os.path.exists(model_dir) else ''

# print("Device:", device),设备启用的是cuda
# 读取数据集，进行数据处理，读取路径时要注意python语言的转义字符。使用绝对路径时前加r。
#read_csv存在在pandas
pd_train_data = pd.read_csv('train.csv')
pd_train_data['title'] = pd_train_data['title'].fillna('')
pd_train_data['abstract'] = pd_train_data['abstract'].fillna('')
# fillna（）：缺失值以" "填充
#处理获取所有测试集、训练集的数据。进行连接
test_data = pd.read_csv('test.csv')
test_data['title'] = test_data['title'].fillna('')
test_data['abstract'] = test_data['abstract'].fillna('')
pd_train_data['text'] = pd_train_data['title'].fillna('') + ' ' +  pd_train_data['author'].fillna('') + ' ' + pd_train_data['abstract'].fillna('')+ ' ' + pd_train_data['Keywords'].fillna('')
test_data['text'] = test_data['title'].fillna('') + ' ' +  test_data['author'].fillna('') + ' ' + test_data['abstract'].fillna('')+ ' ' + pd_train_data['Keywords'].fillna('')
validation_data = pd_train_data.sample(frac=validation_ratio)
train_data = pd_train_data[~pd_train_data.index.isin(validation_data.index)]


class MyDataset(Dataset):

    def __init__(self, mode='train'):
        super(MyDataset, self).__init__()
        self.mode = mode
        # 拿到对应的数据
        if mode == 'train':
            self.dataset = train_data
        elif mode == 'validation':
            self.dataset = validation_data
        elif mode == 'test':
            # 如果是测试模式，则返回内容和uuid。拿uuid做target主要是方便后面写入结果。
            self.dataset = test_data
        else:
            raise Exception("Unknown mode {}".format(mode))

    def __getitem__(self, index):
        # 取第index条
        data = self.dataset.iloc[index]
        # 取其内容
        text = data['text']
        # 根据状态返回内容
        if self.mode == 'test':
            # 如果是test，将uuid做为target
            label = data['uuid']
        else:
            label = data['label']
        # 返回内容和label
        return text, label

    def __len__(self):
        return len(self.dataset)


train_dataset = MyDataset('train')
validation_dataset = MyDataset('validation')
# 获取Bert预训练模型
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# 构造dataloader，定义一下collate_fn，在其中完成对句子进行编码、填充、组装batch等动作
def collate_fn(batch):
    """
    将一个batch的文本句子转成tensor，并组成batch。
    :param batch: 一个batch的句子，例如: [('推文', target), ('推文', target), ...]
    :return: 处理后的结果，例如：
             src: {'input_ids': tensor([[ 101, ..., 102, 0, 0, ...], ...]), 'attention_mask': tensor([[1, ..., 1, 0, ...], ...])}
             target：[1, 1, 0, ...]
    """
    text, label = zip(*batch)
    text, label = list(text), list(label)

    # src是要送给bert的，所以不需要特殊处理，直接用tokenizer的结果即可
    # padding='max_length' 不够长度的进行填充
    # truncation=True 长度过长的进行裁剪
    src = tokenizer(text, padding='max_length', max_length=text_max_length, return_tensors='pt', truncation=True)

    return src, torch.LongTensor(label)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
inputs, targets = next(iter(train_loader))
class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()

        # 加载bert模型
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # 最后的预测层
        self.predictor = nn.Sequential(
            nn.Linear(768, 256),
            #添加relu防止过拟合
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, src):
        """
        :param src: 分词后的推文数据
        """

        # 将src直接序列解包传入bert，因为bert和tokenizer是一套的，所以可以这么做。
        # 得到encoder的输出，用最前面[CLS]的输出作为最终线性层的输入
        outputs = self.bert(**src).last_hidden_state[:, 0, :]

        # 使用线性层来做最终的预测
        return self.predictor(outputs)

model = MyModel()
model = model.to(device)
#定义出损失函数和优化器。这里使用Binary Cross Entropy：
criteria = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# 由于inputs是字典类型的，定义一个辅助函数帮助to(device)
def to_device(dict_tensors):
    result_tensors = {}
    for key, value in dict_tensors.items():
        result_tensors[key] = value.to(device)
    return result_tensors
#定义一个验证方法，获取到验证集的精准率和loss。
def validate():
#启动模型验证方式
    model.eval()
    total_loss = 0.
    total_correct = 0
    for inputs, targets in validation_loader:
        inputs, targets = to_device(inputs), targets.to(device)
        outputs = model(inputs)
        loss = criteria(outputs.view(-1), targets.float())
        total_loss += float(loss)

        correct_num = (((outputs >= 0.5).float() * 1).flatten() == targets).sum()
        total_correct += correct_num

    return total_correct / len(validation_dataset), total_loss / len(validation_dataset)

#训练部分
# 首先将模型调成训练模式
model.train()

# 清空一下cuda缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 定义几个变量，帮助打印loss
total_loss = 0.
# 记录步数
step = 0

# 记录在验证集上最好的准确率
best_accuracy = 0

# 开始训练
for epoch in range(epochs):
    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        # 从batch中拿到训练数据
        inputs, targets = to_device(inputs), targets.to(device)
        # 传入模型进行前向传递
        outputs = model(inputs)
        # 计算损失
        loss = criteria(outputs.view(-1), targets.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += float(loss)
        step += 1

        if step % log_per_step == 0:
            print("Epoch {}/{}, Step: {}/{}, total loss:{:.4f}".format(epoch+1, epochs, i, len(train_loader), total_loss))
            total_loss = 0

        del inputs, targets

    # 一个epoch后，使用过验证集进行验证
    accuracy, validation_loss = validate()
    print("Epoch {}, accuracy: {:.4f}, validation loss: {:.4f}".format(epoch+1, accuracy, validation_loss))
    torch.save(model, model_dir / f"model_{epoch}.pt")

    # 保存最好的模型
    if accuracy > best_accuracy:
        torch.save(model, model_dir / f"model_best.pt")
        best_accuracy = accuracy
#加载好的模型进行测试集预测
model = torch.load(model_dir / f"model_best.pt")
model = model.eval()

test_dataset = MyDataset('test')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
#将测试数据送入模型进行好的预测
results = []
for inputs, ids in test_loader:
    outputs = model(inputs.to(device))
    outputs = (outputs >= 0.5).int().flatten().tolist()
    ids = ids.tolist()
    results = results + [(id, result) for result, id in zip(outputs, ids)]
test_label = [pair[1] for pair in results]
test_data['label'] = test_label
test_data[['uuid', 'Keywords', 'label']].to_csv('submit_task11.csv', index=None)
# 导入pandas用于读取表格数据
import pandas as pd

# 导入BOW（词袋模型），可以选择将CountVectorizer替换为TfidfVectorizer（TF-IDF（词频-逆文档频率）），注意上下文要同时修改，亲测后者效果更佳
from sklearn.feature_extraction.text import TfidfVectorizer
# 导入Bert模型
from sentence_transformers import SentenceTransformer

# 导入计算相似度前置库，为了计算候选者和文档之间的相似度，我们将使用向量之间的余弦相似度，因为它在高维度下表现得相当好。
from sklearn.metrics.pairwise import cosine_similarity

# 过滤警告消息
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
import warnings
warnings.filterwarnings("ignore")
# 读取数据集
test = pd.read_csv('testB.csv')
test['title'] = test['title'].fillna('')
test['abstract'] = test['abstract'].fillna('')
test['text'] = test['title'].fillna('') + ' ' + test['abstract'].fillna('')

# 定义停用词，去掉出现较多，但对文章不关键的词语
stops = [i.strip() for i in open(r'stop.txt', encoding='utf-8').readlines()]

# 这里我们使用distiluse-base-multilingual-cased，因为它在相似性任务中表现出了很好的性能，这也是我们对关键词/关键短语提取的目标!
# 由于transformer模型有token长度限制，所以在输入大型文档时，你可能会遇到一些错误。在这种情况下，您可以考虑将您的文档分割成几个小的段落，并对其产生的向量进行平均池化（mean pooling ，要取平均值）。

model = SentenceTransformer(r'xlm-r-distilroberta-base-paraphrase-v1')
test_words = []
for row in test.iterrows():
    # 读取第每一行数据的标题与摘要并提取关键词

    n_gram_range = (2, 2)
    # 这里我们使用TF-IDF算法来获取候选关键词
    count = TfidfVectorizer(ngram_range=n_gram_range, stop_words=stops).fit([row[1].text])
    candidates = count.get_feature_names_out()
    # 将文本标题以及候选关键词/关键短语转换为数值型数据（numerical data）。我们使用BERT来实现这一目的
    title_embedding = model.encode([row[1].title])

    candidate_embeddings = model.encode(candidates)

    # 通过修改这个参数来更改关键词数量
    top_n = 15
    # 利用文章标题进一步提取关键词
    distances = cosine_similarity(title_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

    if len(keywords) == 0:
        keywords = ['A', 'B']
    test_words.append('; '.join(keywords))
# 输出
# test_label = [pair[1] for pair in test_words]
# test['label'] = test_label
test['Keywords'] = test_words
test[['uuid', 'Keywords','label']].to_csv('submit_task2.csv', index=None)