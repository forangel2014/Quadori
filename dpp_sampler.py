import torch
import numpy as np
from dpp.dpp import dpp
from transformers import BertModel, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer, RobertaTokenizer, RobertaModel
import torch
from sklearn.cluster import KMeans, AgglomerativeClustering
from sentence_transformers import SentenceTransformer

class DPPsampler():

    def __init__(self, device, model_dir=None):

        self.device = device
        
        
        self.model_name = 'bert-base-uncased' if model_dir is None else model_dir
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name).eval().cuda(self.device)
        
        '''
        self.model_name = 'roberta-base' if model_dir is None else model_dir
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.model = RobertaModel.from_pretrained(self.model_name).eval().cuda(self.device)
        '''
        
        # self.rescorer_name = "gpt2"
        # self.rescorer_tokenizer = GPT2Tokenizer.from_pretrained(self.rescorer_name)
        # self.rescorer = GPT2LMHeadModel.from_pretrained(self.rescorer_name).eval().cuda(self.device)

    def tokenize(self, sents, tokenizer):
        ids = []
        max_len = 0
        for sent in sents:
            id = self.tokenizer.encode(sent[0])
            ids.append(id)
            if len(id) > max_len:
                max_len = len(id)
        for id in ids:
            id.extend([tokenizer.pad_token_id]*(max_len-len(id)))
        return ids

    def get_repr(self, sents):
        ids = self.tokenize(sents, self.tokenizer)
        ids = torch.tensor(ids).cuda(self.device)
        repr = self.model(ids)[0]
        repr = torch.mean(repr, dim=1)
        return repr

    # def Kmeans_clusting(self, sents, n_clusters=5):
    #     repr = self.get_repr(sents)
    #     kmeans = KMeans(n_clusters=n_clusters)
    #     result = kmeans.fit(repr.detach().cpu().numpy())
    #     return result.labels_

    # def Hierarchical_clusting(self, sents, n_clusters=5):
    #     repr = self.get_repr(sents)
    #     clustering = AgglomerativeClustering(n_clusters=n_clusters)
    #     result = clustering.fit(repr.detach().cpu().numpy())
    #     return result.labels_

    def get_L(self, sents):
        repr_raw = self.get_repr(sents)
        repr_norm = repr_raw/torch.norm(repr_raw, dim=1, keepdim=True)
        #scores = torch.softmax(torch.tensor([sent[1] for sent in sents]).cuda(self.device), axis=0)
        scores = torch.tensor([sent[1] for sent in sents]).cuda(self.device)
        #scores = torch.tensor([1 for sent in sents]).cuda(self.device)
        repr = torch.matmul(torch.diag(scores.to(repr_norm.dtype)), repr_norm)
        L_raw = torch.matmul(repr, repr.T)
        L_diag = torch.diag(scores-L_raw.diag())
        L = L_raw + L_diag
        #print(L)
        return L

    def dpp(self, sents, k):
        L = self.get_L(sents)
        lam, v = torch.linalg.eigh(L)
        #invlam = lam/(1+lam)
        K = torch.matmul(torch.linalg.inv(L+torch.eye(lam.shape[0]).cuda(self.device)), L)
        selected_ids = dpp(K.detach().cpu().numpy(), max_length=k)
        return selected_ids

    # def rescoring(self, rhs_ls):
    #     scores = []
    #     for r in rhs_ls:
    #         sentence = r[0]
    #         inputs = self.rescorer_tokenizer.encode(sentence, return_tensors='pt').cuda(self.device)
    #         score = self.rescorer(inputs, labels=inputs)[0]
    #         scores.append(np.exp(-score.tolist()))
    #         #r.append(score)
    #     return scores

class NewDPPsampler():

    def __init__(self, device, model_name='all-MiniLM-L6-v2'):
        self.device = device
        self.model = SentenceTransformer(model_name).to(device)

    def tokenize(self, sents, tokenizer):
        # 使用tokenizer的padding参数
        encoded_inputs = tokenizer(sents, padding=True, return_tensors='pt')
        return encoded_inputs['input_ids'].to(self.device)

    def get_repr(self, sents):
        # 使用更好的句子表示模型
        sents = [sent[0] for sent in sents]
        repr = self.model.encode(sents, convert_to_tensor=True, device=self.device)
        return repr

    def get_L(self, sents):
        repr_raw = self.get_repr(sents)
        repr_norm = repr_raw/torch.norm(repr_raw, dim=1, keepdim=True)
        #scores = torch.softmax(torch.tensor([sent[1] for sent in sents]).cuda(self.device), axis=0)
        scores = torch.tensor([sent[1] for sent in sents]).cuda(self.device)
        #scores = torch.tensor([1 for sent in sents]).cuda(self.device)
        repr = torch.matmul(torch.diag(scores.to(repr_norm.dtype)), repr_norm)
        L_raw = torch.matmul(repr, repr.T)
        L_diag = torch.diag(scores-L_raw.diag())
        L = L_raw + L_diag
        #print(L)
        return L

    def dpp(self, sents, k):
        L = self.get_L(sents)
        lam, v = torch.linalg.eigh(L)
        #invlam = lam/(1+lam)
        K = torch.matmul(torch.linalg.inv(L+torch.eye(lam.shape[0]).cuda(self.device)), L)
        selected_ids = dpp(K.detach().cpu().numpy(), max_length=k)
        return selected_ids

    # 其他方法可以根据需要添加

# if __name__ == "__main__":
#     sampler = DPPsampler(0)
#     res = sampler.Hierarchical_clusting([["participate in", 1], ["will participate in", 1], ["has participated in", 1], ["how are you", 1]], 2)
#     print(res)
