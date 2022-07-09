from inductor import BartInductor

inductor = BartInductor()

rule = '<mask> are geographically ditributed in <mask>.'
generated_texts = inductor.generate(rule, k=10, topk=10)

print('output generated rules:')
for text in generated_texts:
    print(text)
