from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from transformers import BertForMaskedLM  

model_neuralmind = BertForMaskedLM.from_pretrained('neuralmind/bert-base-portuguese-cased', output_loading_info=False)
tokenizer_neuralmind = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False, output_loading_info=False)

pipe = pipeline('fill-mask', model=model_neuralmind, tokenizer=tokenizer_neuralmind)
output = pipe('Tinha uma [MASK] no meio do caminho.')
print(output)
