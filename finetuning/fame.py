import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import wandb
from peft import PeftModel, get_peft_config, get_peft_model, LoraConfig, TaskType
from transformers import DataCollatorForLanguageModeling
from huggingface_hub import HfApi, create_repo
import os
import shutil

# Configuration variables
BASE_MODEL_NAME = "subhrokomol/Meta-Llama-3-8B-Hindi"
DATASET_NAME = "zicsx/mC4-Hindi-Cleaned-3.0"
DATASET_PERCENTAGE = "0.003%"
OUTPUT_DIR_NAME = "fine_tuned_llama"
FULL_OUTPUT_DIR_NAME = "full_fine_tuned_llama"
HF_MODEL_NAME = "subhrokomol/fine-tuned-llama-3-hindi"
UPLOAD_TO_HF = True
SAVE_FULL_MODEL = True  # Set to False to skip saving full model files

# Color codes for debug messages
BLUE = "\033[34m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"

def debug_print(message, color=BLUE):
    print(f"{color}DEBUG:{RESET} {message}")

def skip_training():
    return False

# Initialize wandb
wandb.init(project="llama-3-finetuning")

debug_print(f"Loading tokenizer from: {BASE_MODEL_NAME}", GREEN)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
debug_print("Tokenizer loaded successfully.", GREEN)

debug_print(f"Loading model from: {BASE_MODEL_NAME}", GREEN)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16)
debug_print("Model loaded successfully.", GREEN)

model.gradient_checkpointing_enable()

# Define LoRA Configuration
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        # "gate_proj",
        # "up_proj",
        # "down_proj",
    ]
)

# Get PEFT model
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

debug_print("Loading dataset...", YELLOW)
dataset = load_dataset("zicsx/mC4-Hindi-Cleaned-3.0", split="train[:3%]")
debug_print("Dataset loaded successfully.", GREEN)

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=256)

dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

debug_print("Starting training...", YELLOW)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR_NAME,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    warmup_steps=500,
    # max_steps=None, #remove entirely or set it to a positive integer when epoch >= 1 
    learning_rate=5e-4,
    fp16=False,
    bf16=True,
    logging_steps=50,
    save_steps=1000,
    save_strategy="steps", 
    save_total_limit=1,
    max_grad_norm=1.0,
    eval_steps=500,
    report_to="wandb",
    run_name="llama-3-8b-finetuning",
    remove_unused_columns=False,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Train
if not skip_training():
    trainer.train()
    debug_print(f"Saving the fine-tuned model to {OUTPUT_DIR_NAME}...", GREEN)
    trainer.save_model(OUTPUT_DIR_NAME)
else:
    debug_print(f"Skipping training, using existing fine-tuned model in {OUTPUT_DIR_NAME}.", YELLOW)

# Close the wandb run
# After training (remove the trainer.save_model call)

# Close the wandb run
wandb.finish()

debug_print("Loading the base model...", YELLOW)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.bfloat16)

debug_print(f"Loading the LoRA weights from {OUTPUT_DIR_NAME}...", YELLOW)
peft_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR_NAME)

debug_print("Merging LoRA weights with base model...", YELLOW)
merged_model = peft_model.merge_and_unload()

if SAVE_FULL_MODEL:
    debug_print(f"Saving full merged model locally to {FULL_OUTPUT_DIR_NAME}...", GREEN)
    merged_model.save_pretrained(FULL_OUTPUT_DIR_NAME, safe_serialization=True)
    tokenizer.save_pretrained(FULL_OUTPUT_DIR_NAME)

    # No need to generate pytorch_model.bin separately, it's included in save_pretrained

if UPLOAD_TO_HF:
    debug_print("Preparing to upload to Hugging Face Hub...", YELLOW)
    api = HfApi()

    try:
        debug_print(f"Creating repository: {HF_MODEL_NAME}", YELLOW)
        create_repo(HF_MODEL_NAME, private=False)
    except Exception as e:
        debug_print(f"Repository creation failed, it might already exist: {e}", RED)

    debug_print(f"Uploading model to {HF_MODEL_NAME}", YELLOW)
    api.upload_folder(
        folder_path=FULL_OUTPUT_DIR_NAME,
        repo_id=HF_MODEL_NAME,
        repo_type="model"
    )

    debug_print("Model successfully uploaded to Hugging Face Hub!", GREEN)
else:
    debug_print("Skipping upload to Hugging Face Hub as per configuration.", YELLOW)

debug_print(f"Process completed. Full model saved locally in {FULL_OUTPUT_DIR_NAME}", GREEN)