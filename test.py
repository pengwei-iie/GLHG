import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('/data/pretrained_models/comet-distill-tokenizer')
model = GPT2LMHeadModel.from_pretrained('/data/pretrained_models/comet-distill-low')
inp = tokenizer ("<head> PersonX eats an apple . </head> <relation> xWant </relation> [GEN] ") ['input_ids']
gen = tokenizer.decode(model.generate(torch.tensor(inp).view(1,-1)).tolist()[0])
print(gen)
