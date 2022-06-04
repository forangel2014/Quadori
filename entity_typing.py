import pickle
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertModel, BertTokenizer

class EntityTpingDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ids = self.data[index][0]
        label = self.data[index][1]
        return ids, label

def get_collate_fn(pad_token_id=0):
    def collate_fn(batch):
        max_len = 0
        for data in batch:
            if len(data[0]) > max_len:
                max_len = len(data[0])
        ids = []
        labels = []
        masks = []
        for data in batch:
            n = len(data[0])
            ids.append(data[0] + [pad_token_id]*(max_len-n))
            masks.append([1]*n + [pad_token_id]*(max_len-n))
            labels.append(data[1])
        return ids, masks, labels
    return collate_fn

class BertEntityTyper(torch.nn.Module):

    def __init__(self, model_name, n_labels):
        super(BertEntityTyper, self).__init__()
        self.model_name = model_name
        self.n_labels = n_labels
        self.config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)

        self.linear = torch.nn.Linear(self.config.hidden_size, n_labels)

    def forward(self, ids, masks):
        output = self.bert(input_ids=ids, attention_mask=masks)[0]
        probs = torch.softmax(self.linear(output[:, 0, :]), dim=-1)
        return probs

class EntityTyper():

    def __init__(self, data, model_name, device, batch_size, lr, epoch):
        self.data = data
        self.model_name = model_name 
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.epoch = epoch

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.dataset = EntityTpingDataset(data)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, collate_fn=get_collate_fn(self.tokenizer.pad_token_id))
        self.model = BertEntityTyper(model_name, len(labels)).cuda(device)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        for e in range(epoch):
            for i, (ids, masks, labels) in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                ids = torch.tensor(ids).cuda(device)
                masks = torch.tensor(masks).cuda(device)
                labels = torch.tensor(labels).cuda(device)
                probs = self.model(ids, masks)
                loss = self.criterion(probs, labels)
                loss.backward()
                self.optimizer.step()
                print(loss)
                

model_name = 'bert-base-uncased'
device = 'cuda:8'
batch_size = 10
lr = 1e-5
epoch = 5

datadir = '/data/sunwangtao/Orion/data/wiki/wiki_def/'
labelfile = open(datadir + 'label.pkl', 'rb') 
datafile = open(datadir + 'data.pkl', 'rb')
labels = pickle.load(labelfile)
data = pickle.load(datafile)

typer = EntityTyper(data, model_name, device, batch_size, lr, epoch)
typer.train()