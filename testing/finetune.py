import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer

# Load the dataset
dataset = load_dataset("zicsx/mC4-Hindi-Cleaned-3.0", split="train")

# Model and tokenizer
model_id = "subhrokomol/llama2-7b-english-hindi"

# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

# Prepare the model for k-bit training
model = prepare_model_for_kbit_training(model)

# LoRA configuration
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Get the PEFT model
model = get_peft_model(model, peft_config)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Training arguments
from transformers import TrainingArguments

training_arguments = TrainingArguments(
    output_dir="./results_hindi_llama",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=500,
    logging_steps=100,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=2000,  # Adjust this based on your time constraint
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)

# Prepare the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
)

# Start training
trainer.train()

# Save the model
trainer.model.save_pretrained("./hindi_llama_finetuned")