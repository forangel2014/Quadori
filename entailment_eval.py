import torch
from transformers import BartForSequenceClassification, BartTokenizer, RobertaForSequenceClassification, RobertaTokenizer

def postprocess(r):
    return r.replace('<mask>', 'A', 1).replace('<mask>', 'B', 1)

class EntailmentScorer():
    
    def __init__(self, device):
        self.device = device
        self.model_path = "roberta-large-mnli" #"geckos/bart-fined-tuned-on-entailment-classification"
        #self.model = BartForSequenceClassification.from_pretrained(self.model_path).eval().cuda(device)
        #self.tokenizer = BartTokenizer.from_pretrained(self.model_path)
        
        self.model = RobertaForSequenceClassification.from_pretrained(self.model_path).eval().cuda(device)
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_path)
        
    
    def scoring(self, r_p, r_h):
        sentence = postprocess(r_p) + ' ' + postprocess(r_h)
        ids = self.tokenizer.encode(sentence, return_tensors='pt').cuda(self.device)
        res = self.model(ids) # [contradiction, neutral, entailment]
        probs = torch.softmax(res[0], dim=-1)
        return probs[0][2].tolist()
    

if __name__ == "__main__":
    scorer = EntailmentScorer("cuda:0")
    res = scorer.scoring("I like you.", "I love you")
    print(res)