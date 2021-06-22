from transformers import pipeline

pipe = pipeline('fill-mask', model=model_neuralmind, tokenizer=tokenizer_neuralmind)
output = pipe('Tinha uma [MASK] no meio do caminho.')
print(output)
