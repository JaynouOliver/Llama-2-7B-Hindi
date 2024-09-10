import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np

def calculate_perplexity(model, tokenizer, texts, max_length=1024):
    model.eval()
    total_loss = 0
    total_length = 0

    with torch.no_grad():
        for text in texts:
            encodings = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True)
            input_ids = encodings.input_ids.to(model.device)
            target_ids = input_ids.clone()
            
            outputs = model(input_ids, labels=target_ids)
            loss = outputs.loss
            
            total_loss += loss.item() * input_ids.size(1)
            total_length += input_ids.size(1)

    perplexity = torch.exp(torch.tensor(total_loss / total_length))
    return perplexity.item()

# Load models
base_model_name = "subhrokomol/llama2-7b-english-hindi"
finetuned_model_name = "subhrokomol/Llama2-7B-Hindi-finetuned"

base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16, device_map="auto")
base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)

finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_model_name, torch_dtype=torch.float16, device_map="auto")
finetuned_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_name)

# Load a Hindi dataset for evaluation
dataset = load_dataset("oscar", "unshuffled_deduplicated_hi", split="train[:1000]")  # Using 1000 samples

# Calculate perplexity
base_perplexity = calculate_perplexity(base_model, base_tokenizer, dataset['text'])
finetuned_perplexity = calculate_perplexity(finetuned_model, finetuned_tokenizer, dataset['text'])

print(f"Base Model Perplexity: {base_perplexity}")
print(f"Fine-tuned Model Perplexity: {finetuned_perplexity}")