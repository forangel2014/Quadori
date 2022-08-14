from inductor import BartInductor

inductor = BartInductor(prompt=True, mcgs=True, dpp=True)

rule = '<mask> marries <mask>.'
#rule = '<mask> is the cause of disease <mask>.'
#rule = '<mask> is a part of <mask>.'
generated_texts = inductor.generate(rule, k=20, topk=20)

with open('./data/exp/ours_spouse_20.txt', 'w') as f:
    print('output generated rules:')
    for text in generated_texts:
        print(text)
        f.writelines(text + '\n')
