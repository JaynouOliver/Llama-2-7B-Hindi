# 🚀 Llama-3 Trans-Tokenization and Fine-Tuning Guide

This repository provides a comprehensive guide to trans-tokenization and fine-tuning Llama-3 models. The primary objective is to achieve optimal performance in translating and training large language models (LLMs) across different languages.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Trans-Tokenization](#trans-tokenization)
- [Fine-Tuning](#fine-tuning)
- [Dataset](#dataset)
- [Time Estimation](#time-estimation)
- [Supported Tools](#supported-tools)
- [Understanding Tokenization](#understanding-tokenization)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 📚 Overview

This guide covers:
1. Setting up the base model and tokenizer.
2. Implementing trans-tokenization.
3. Fine-tuning the Llama-3 model.
4. Understanding tokenization challenges with Latin and non-Latin scripts.

---

## 🛠️ Installation

### Clone the Repository


git clone https://github.com/FremyCompany/fast_align
mkdir build
cd build
cmake ..
make
cd .. # return to the original folder

### 2. Create a Conda Environment

```
conda create -n llama-env python=3.10
conda activate llama-env
pip install -r requirements.txt
```

## 🔄 Trans-Tokenization
To perform trans-tokenization, you will need two key files:

transtokenization.py
run.py
Example Setup in run.py:

```
source_model = "meta-llama/Meta-Llama-3-8B"
target_tokenizer = "yhavinga/gpt-neo-1.3B-dutch"
export_dir = "en-nl-llama3-8b"

corpus = create_aligned_corpus(
    source_language="en",
    target_language="nl",
    source_tokenizer=source_model,
    target_tokenizer=target_tokenizer,
)
```

### Supported Languages and Datasets
You can view the list of supported languages in the CCMATRIX_MAPPING section of transtokenizers.py.

Select the dataset from the corpus_list:
```
corpus_list = ["allenai/nllb", ]
```

### Currently Supported Datasets:
```
open_subtitles
allenai/nllb
```

### Running the Script
After setting up the configurations, run the run.py script in your conda environment:
```
python run.py
```
This script will:

Import necessary functions from transtokenizers.py.
Automatically iterate through the data to create the aligned corpus.
Align tokens using Fast Align.
Smooth and remap the model.
The final output model will be saved in your specified export directory.

### ⚠️ Storage Considerations
If you are short on storage, you can stop the script after the dataset download begins. This will create a new folder with the partially downloaded dataset. Upon re-running the script, it will handle the edge case and continue from the next step.

### Troubleshooting
If the token mapping reaches 100%, there may be an issue with your code. Check the Moses file and the TSV file generated after the process for potential errors.

### Translation Performance
We achieved a translation accuracy of 87% on the Hindi dataset. You can fine-tune this model further to achieve better results.


## 🎛️ Fine-Tuning
Setting Up
1. Create a new conda environment:
    ```
    conda create -n finetune-env python=3.10
    conda activate finetune-env
    pip install -r requirements.txt
    ```
2. Login to Hugging Face and Weights & Biases:
   ```
   huggingface-cli login
   wandb login
   ```
#### Configurations in finetune.py:
```
BASE_MODEL_NAME = "subhrokomol/Meta-Llama-3-8B-Hindi"
DATASET_NAME = "zicsx/mC4-Hindi-Cleaned-3.0"
OUTPUT_DIR_NAME = "fine_tuned_llama"
HF_MODEL_NAME = "subhrokomol/fine-tuned-llama-3-hindi"
UPLOAD_TO_HF = True
SAVE_FULL_MODEL = True

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16
)

training_args = TrainingArguments( #chenage them as you prefer
    output_dir=OUTPUT_DIR_NAME,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    warmup_steps=500,
    learning_rate=5e-4,
    bf16=True,
    logging_steps=50,
    save_steps=1000,
    eval_steps=500,
    report_to="wandb",
    run_name="llama-3-8b-finetuning",
)
```
3. Start fine-tuning:
    python finetune.py
4. The final model will be uploaded to your Hugging Face repository.

# Training Time Estimates for Dataset

## Dataset Information
- **Size**: 17 GB
- **Number of Rows**: 4 Million

## Model Specifications
- **Parameters**: 8 Billion

## Estimated Training Time

| Dataset Size         | Model Parameters | Estimated Time |
|----------------------|------------------|----------------|
| 17 GB (3% of dataset) | 8B              | 6 hours        |
| 17 GB (7% of dataset) | 8B              | 12 hours       |
| 17 GB (50% of dataset)| 8B              | 40 hours       |

**Note:** The estimated times are based on training with the specified model parameters and may vary depending on hardware and other factors.

## ⏱️ Time Estimation
3% Dataset: ~6 hours for 1 epoch
7% Dataset: ~12 hours for 1 epoch
50% Dataset: ~40 hours (approx)

## Supported Tools

| Tool                  | Link          |
|-----------------------|---------------|
| Axolotl               | [GitHub](https://github.com/axolotl) |
| Hugging Face PEFT     | [GitHub](https://github.com/huggingface/peft) |
| PyTorch Torchtune     | [GitHub](https://github.com/pytorch/torchtune) |


🧠 Understanding Tokenization
Training a BPE SentencePiece tokenizer is straightforward. You can use the following example to convert the Hugging Face format:
```
def convert_to_hf_format(output_path):
    transformers.LlamaTokenizerFast(vocab_file=output_path+'.model').save_pretrained(output_path+'/')
```
Challenges
Latin scripts tend to have a higher percentage of tokens compared to non-Latin scripts. This can affect translation accuracy.

![newplot](https://github.com/user-attachments/assets/f02c70da-f162-40ae-8f28-b71cdb482f21)
Image showing hindi and english tokenization with Llama 3 8B 
https://huggingface.co/spaces/yenniejun/tokenizers-languages#median-token-length-for-openai-gpt4


🌟 Contributing
We welcome contributions from the community! Please read the Contributing Guidelines for more information.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

