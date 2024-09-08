from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "new_model"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Set a custom chat template
tokenizer.chat_template = """{% for message in messages %}
{% if message['role'] == 'system' %}System: {{ message['content'] }}
{% elif message['role'] == 'user' %}Human: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}
{% endif %}
{% endfor %}
Human: {{ messages[-1]['content'] }}
Assistant:"""

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

# Apply chat template and create input_ids
inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt"
)

# Move inputs to the same device as the model
inputs = inputs.to(model.device)

# Create attention mask
attention_mask = torch.ones_like(inputs)

# Set pad_token_id
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

outputs = model.generate(
    inputs,
    attention_mask=attention_mask,
    max_new_tokens=256,
    pad_token_id=tokenizer.pad_token_id,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

response = outputs[0][inputs.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))