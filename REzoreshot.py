import re
import tqdm
from inductor import BartInductor

inductor = BartInductor(prompt=False, ssts=True, dpp=True, device='cuda:6')

def relation_text_preprocess(text):
    text = text.strip("\"")
    text = re.sub('<e1>.*<\/e1>', '<mask>', text)
    text = re.sub('<e2>.*<\/e2>', '<mask>', text)
    return text

def get_relations(relation_file):
    all_relations = []
    with open(relation_file) as f:
        lines = f.readlines()
        for line in lines:
            id, r_tag, r_text = line[:-1].split(' ', 2)
            all_relations.append((r_tag, r_text))
    return all_relations

def load_dataset(data_file):
    dataset = []
    with open(data_file) as f:
        lines = f.readlines()
        for i in range(len(lines)//4):
            text = lines[4*i][:-1].split('\t')[1]
            text = relation_text_preprocess(text)
            tag = lines[4*i+1][:-1]
            dataset.append((text, tag))
    return dataset

def get_all_ins(all_relations):
    all_relation_ins = []
    for r_tag, r_text in all_relations:
        ins = inductor.generate_ins_from_hypo(r_text)
        all_relation_ins.append((r_tag, r_text, ins))
    return all_relation_ins

datadir = "../datasets/Semeval/"
relation_file = datadir+"relations.txt"
train_file = datadir+"TRAIN_FILE.TXT"
all_relations = get_relations(relation_file)
all_relation_ins = get_all_ins(all_relations)
train_set = load_dataset(train_file)[:100]
right_num = 0
for text, tag in tqdm.tqdm(train_set):
    score_tag = []
    for r_tag, r_text, ins in all_relation_ins:
        score = inductor.score_premise_through_ins(ins, r_text, text).tolist()
        score_tag.append((score, r_tag))
    score_tag.sort(key=lambda x:x[0])
    tag_predict = score_tag[0][1]
    print(score_tag)
    print(tag)
    right_num += tag_predict == tag
print(right_num/len(train_set))