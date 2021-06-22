from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from transformers import BertForMaskedLM  

model_neuralmind = BertForMaskedLM.from_pretrained('neuralmind/bert-base-portuguese-cased')
tokenizer_neuralmind = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False)
