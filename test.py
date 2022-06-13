from inductor import BartInductor

inductor = BartInductor()

rule = '<mask> is geographically distributed in <mask>.'
#rule = 'Those invited included the Republican fund-raiser Georgette Mosbacher and the former chief executives Leonard A. Lauder -LRB- Estée Lauder Companies -RRB- and <mask> -LRB- <mask> -RRB- , though it was not clear if all those invited attended . '''
generated_texts = inductor.generate(rule, k=10, topk=10)

print('output generated rules:')
for text in generated_texts:
    print(text)
