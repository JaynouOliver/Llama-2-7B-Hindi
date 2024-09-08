import os
import torch
from transformers import GPTNeoForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, GPTNeoConfig, PreTrainedTokenizerFast
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset

# Load the Hindi Wikipedia dataset
dataset = load_dataset("wikimedia/wikipedia", "20231101.hi")

# Function to combine title and text
def combine_title_text(example):
    return {"content": example["title"] + " " + example["text"]}

# Apply the function to the dataset
dataset = dataset.map(combine_title_text)

# Keep only the 'content' column
dataset = dataset.remove_columns(["id", "url", "title", "text"])

# Split the dataset into train and validation sets
train_val = dataset["train"].train_test_split(test_size=0.1)
train_data = train_val["train"]
val_data = train_val["test"]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Prepare the training data
def batch_iterator():
    for i in range(0, len(train_data), 1000):
        yield train_data[i:i+1000]["content"]

# Train the tokenizer
tokenizer.train_from_iterator(
    batch_iterator(),
    vocab_size=52000,
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)

# Create the directory if it doesn't exist
os.makedirs("hindi_tokenizer", exist_ok=True)

# Save the tokenizer
tokenizer.save_model("hindi_tokenizer")

# Train the tokenizer as before
tokenizer = ByteLevelBPETokenizer()

def batch_iterator():
    for i in range(0, len(train_data), 1000):
        yield train_data[i:i+1000]["content"]

tokenizer.train_from_iterator(
    batch_iterator(),
    vocab_size=52000,
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)

# Convert to PreTrainedTokenizerFast
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

# Add necessary tokens and settings
fast_tokenizer.add_special_tokens({'pad_token': '<pad>'})
fast_tokenizer.bos_token = '<s>'
fast_tokenizer.eos_token = '</s>'
fast_tokenizer.unk_token = '<unk>'
fast_tokenizer.mask_token = '<mask>'

# Save the tokenizer
fast_tokenizer.save_pretrained("./hindi_tokenizer")


config = GPTNeoConfig(
    vocab_size=52000,
    max_position_embeddings=2048,
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    intermediate_size=3072,
    attention_types=[[["global", "local"], 6]],
    window_size=256,
)

config.save_pretrained("./gpt-neo-hindi-config")

#training
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("./hindi_tokenizer")
tokenizer.add_special_tokens({'pad_token': '<pad>'})

# Load the model configuration
config = GPTNeoConfig.from_pretrained("./gpt-neo-hindi-config")
model = GPTNeoForCausalLM(config)
model.resize_token_embeddings(len(tokenizer))

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["content"], truncation=True, max_length=512, padding="max_length")

tokenized_train = train_data.map(tokenize_function, batched=True, remove_columns=train_data.column_names)
tokenized_val = val_data.map(tokenize_function, batched=True, remove_columns=val_data.column_names)

# Set up the trainer
training_args = TrainingArguments(
    output_dir="./gpt-neo-hindi",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_steps=5000,
    save_steps=5000,
    warmup_steps=1000,
    prediction_loss_only=True,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
)

# Start training
trainer.train()

# Save the model
trainer.save_model("./gpt-neo-hindi-final")