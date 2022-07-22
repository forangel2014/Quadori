import re
import torch
from torch.optim import Adam
import numpy as np
from dpp_sampler import DPPsampler

def clean_references(texts):
    for i, text in enumerate(texts):
        if text.endswith(" ."):
            texts[i] = text.replace(" .", ".")
    
    return texts

if __name__ == "__main__":
    model_dir = "./model/dpp_encoder/"
    sampler = DPPsampler(0)
    optimizer = Adam(sampler.model.parameters(), lr=1e-6)
    
    ref = []
    with open('./data/OpenRule155.txt', 'r', encoding='utf-8') as file:
        data = file.readlines()
        for row in data:
            row = row.strip().split('\t')
            inputs, head, tail, relations = row[0], row[1], row[2], row[3]
            inputs = inputs.strip()
            
            if relations.startswith('[') and relations.endswith(']'):
                inputs = re.sub("<A>|<B>", "<mask>", inputs)
                references = [relation.replace('<A>', 'A').replace('<B>', 'B').lower().strip() for relation in eval(relations)]
            else:
                references = [relations.replace('[X]', '<mask>').replace('[Y]', '<mask>').lower().strip()]
            references = clean_references(references)
            
            if len(references) > 1:
                ref.append([[r, 1] for r in references])
    
    for e in range(1000):
        batch_size = 10
        for r in ref:
            optimizer.zero_grad()
            L = sampler.get_L(r)
            loss = -torch.log(torch.det(L)/torch.det(L+torch.eye(L.shape[0]).cuda(0)))
            
            loss.backward()
            optimizer.step()
            print(loss)