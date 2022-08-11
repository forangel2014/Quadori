from inductor import BartInductor

inductor = BartInductor(prompt=True, mcgs=True, dpp=True)

#rule = '<mask> marries <mask>.'
rule = '<mask> is geographically distributed in <mask>.'
generated_texts = inductor.generate(rule, k=10, topk=10)

with open('./data/exp/ours_disease_explanation.txt', 'w') as f:
    print('output generated rules:')
    for text in generated_texts:
        print(text)
        f.writelines(text + '\n')
