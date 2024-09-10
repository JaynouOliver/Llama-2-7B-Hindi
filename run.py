from transtokenizers import create_aligned_corpus, align, map_tokens, smooth_mapping, remap_model
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import huggingface_hub as hf_hub

source_model = "subhrokomol/unsloth-mistral7b-en"

target_tokenizer = "subhrokomol/gpt-neo-1.3B-hindi"
export_dir = "model_folder"

corpus = create_aligned_corpus(
    source_language="en",
    target_language="hi",
    source_tokenizer=source_model,
    target_tokenizer=target_tokenizer,
)

mapped_tokens_file = align(corpus, fast_align_path="fast_align/build/fast_align")

tokenized_possible_translations, untokenized_possible_translations = map_tokens(mapped_tokens_file, source_model, target_tokenizer)

smoothed_mapping = smooth_mapping(target_tokenizer, tokenized_possible_translations)

model = remap_model(source_model, target_tokenizer, smoothed_mapping, source_model)
os.makedirs(export_dir, exist_ok=False)
new_tokenizer = AutoTokenizer.from_pretrained(target_tokenizer)
model.save_pretrained(export_dir)
new_tokenizer.save_pretrained(export_dir)

print(f"finished model creation to {export_dir}")
print(f"uploading to HF from {export_dir}")
#upload to HF models
output_model_name = export_dir
repo = hf_hub.create_repo(output_model_name, private=False)  # Set private=False if you want it to be public
hf_hub.upload_folder(
    folder_path=export_dir,
    path_in_repo='',  # Root of the repo
    repo_id=f"{hf_hub.whoami()['name']}/{output_model_name}"
)