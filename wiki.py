import os
import tqdm
from extractor.subject_verb_object_extract import findSVOs, nlp

def extract(str):
    tokens = nlp(str)
    svos = findSVOs(tokens)
    #print(str)
    #print(svos)
    return svos

datadir = './data/wiki/wiki_doc/docs/AB/'
outdir = './data/wiki/wiki_doc/docs/AB_out/'
if not os.path.exists(outdir):
    os.makedirs(outdir)
svo = []
for filename in os.listdir(datadir):
    path = datadir + filename
    print("processing " + path)
    with open(path) as f:
        lines = f.readlines()
        for line in tqdm.tqdm(lines):
            t = extract(line)
            if t is not None:
                svo.extend(t)

    with open(outdir + filename, 'w') as g:
        for t in svo:
            g.writelines('\t'.join(t) + '\n')