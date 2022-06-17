import torch
from transformers import BartForSequenceClassification, BartTokenizer

def postprocess(r):
    return r.replace('<mask>', 'A', 1).replace('<mask>', 'B', 1)

class EntailmentScorer():
    
    def __init__(self, device):
        self.device = device
        self.model = BartForSequenceClassification.from_pretrained("geckos/bart-fined-tuned-on-entailment-classification").eval().cuda(device)
        self.tokenizer = BartTokenizer.from_pretrained("geckos/bart-fined-tuned-on-entailment-classification")
    
    def scoring(self, r_p, r_h):
        sentence = postprocess(r_p) + ' ' + postprocess(r_h)
        ids = self.tokenizer.encode(sentence, return_tensors='pt').cuda(self.device)
        res = self.model(ids) # [contradiction, neutral, entailment]
        probs = torch.softmax(res[0], dim=-1)
        return probs[0][2].tolist()

