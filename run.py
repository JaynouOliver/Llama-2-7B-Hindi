from project import create_aligned_corpus, align, map_tokens, smooth_mapping, remap_model
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

source_model = "meta-llama/Meta-Llama-3-8B"

target_tokenizer = "meta-llama/Meta-Llama-3-8B"
export_dir = "en-hi-llama3-8b"

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