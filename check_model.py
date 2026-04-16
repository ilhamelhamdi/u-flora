from transformers import AutoModel

model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")
print(model)