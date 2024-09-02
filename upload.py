from transformers import LlamaTokenizer, LlamaForCausalLM
from huggingface_hub import HfApi

# Load your model and tokenizer
path = "lora_model"
model = LlamaForCausalLM.from_pretrained(path)
tokenizer = LlamaTokenizer.from_pretrained(path)

# Set up the repository name
repo_name = "subhrokomol/llama-2-7b-bnb-4bit-finetuned-hindi"  # Replace with your desired name

# Push to Hub
model.push_to_hub(repo_name, use_auth_token=True)
tokenizer.push_to_hub(repo_name, use_auth_token=True)

print(f"Model and tokenizer uploaded to {repo_name}")