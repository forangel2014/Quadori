import torch
from dpp.dpp import dpp
from transformers import BertModel, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from sklearn.cluster import KMeans, AgglomerativeClustering

class DPPsampler():

    def __init__(self, device):

        self.device = device
        self.model_name = 'bert-base-uncased'
        self.rescorer_name = "gpt2"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name).eval().cuda(self.device)
        self.rescorer_tokenizer = GPT2Tokenizer.from_pretrained(self.rescorer_name)
        self.rescorer = GPT2LMHeadModel.from_pretrained(self.rescorer_name).eval().cuda(self.device)

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

    def Kmeans_clusting(self, sents, n_clusters=5):
        repr = self.get_repr(sents)
        kmeans = KMeans(n_clusters=n_clusters)
        result = kmeans.fit(repr.detach().cpu().numpy())
        return result.labels_

    def Hierarchical_clusting(self, sents, n_clusters=5):
        repr = self.get_repr(sents)
        kmeans = KMeans(n_clusters=n_clusters)
        result = kmeans.fit(repr.detach().cpu().numpy())
        return result.labels_

    def dpp(self, sents, k):
        repr = self.get_repr(sents)
        repr /= torch.norm(repr, dim=1, keepdim=True)
        scores = torch.tensor([sent[1] for sent in sents]).cuda(self.device)
        repr = torch.matmul(torch.diag(scores.to(repr.dtype)), repr)
        L = torch.matmul(repr, repr.T)
        L_diag = torch.diag(scores-L.diag())
        L += L_diag
        lam, v = torch.linalg.eigh(L)
        #invlam = lam/(1+lam)
        K = torch.matmul(torch.linalg.inv(L+torch.eye(lam.shape[0]).cuda(self.device)), L)
        selected_ids = dpp(K.detach().cpu().numpy(), max_length=k)
        return selected_ids

    def rescoring(self, rhs_ls):
        for r in rhs_ls:
            sentence = r[0]
            inputs = self.rescorer_tokenizer.encode(sentence, return_tensors='pt').cuda(self.device)
            score = self.rescorer(inputs, labels=inputs)[0]
            r.append(score.tolist())
        return rhs_ls