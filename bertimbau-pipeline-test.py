from transformers import AutoModelForTokenClassification, BertForMaskedLM, AutoTokenizer

model_neuralmind = BertForMaskedLM.from_pretrained('neuralmind/bert-base-portuguese-cased')
tokenizer_neuralmind = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False)

from transformers import pipeline

pipe = pipeline('fill-mask', model=model_neuralmind, tokenizer=tokenizer_neuralmind)
output = pipe('Tinha uma [MASK] no meio do caminho.')
print(output)

with open('output.txt', 'w') as f:
    print(output, file=f)

print(pipeline('fill-mask', model=model_neuralmind, tokenizer=tokenizer_neuralmind)('Eu moro em [MASK], Rio Grande do Sul.'))
print(pipeline('fill-mask', model=model_neuralmind, tokenizer=tokenizer_neuralmind)('Eu moro em [MASK] [MASK], Rio Grande do Sul.'))
