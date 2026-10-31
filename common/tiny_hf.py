"""
Tiny, randomly initialised Hugging Face models for tests.

Real checkpoints (Qwen2.5, GPT-2) are hundreds of MB and need network access. The pipelines in this
repository only care that a model *behaves* like a causal LM with a chat template, so tests build a
2-layer Qwen2 with a ~300-token BPE tokenizer in well under a second, entirely offline.
"""

import os

CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

_CORPUS = [
    "<think> 12 plus 30 is 42 </think> <answer> 42 </answer>",
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May.",
    "How many clips did Natalia sell altogether in April and May? The answer is 72.",
    "You are a helpful assistant. Think step by step and give the final answer.",
    "user assistant system 0 1 2 3 4 5 6 7 8 9 10 100 1000 , . ? ! $ % + - * / =",
    "the quick brown fox jumps over the lazy dog because the dog was lazy",
]


def make_tiny_tokenizer(vocab_size=320):
    """BPE tokenizer trained on a few sentences, with ChatML special tokens and chat template."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tok.train_from_iterator(_CORPUS * 4, trainer=trainer)
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    hf_tok.chat_template = CHATML_TEMPLATE
    return hf_tok


def make_tiny_qwen2(vocab_size, seed=0, hidden=32, layers=2, heads=4, kv_heads=2):
    """A 2-layer, 32-wide Qwen2 with random weights (seeded)."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        max_position_embeddings=1024,
        tie_word_embeddings=True,
    )
    return Qwen2ForCausalLM(config)


def save_tiny_chat_model(path, seed=0, **model_kwargs):
    """Write a tiny tokenizer + Qwen2 model to `path` (loadable with from_pretrained); returns `path`."""
    os.makedirs(path, exist_ok=True)
    tok = make_tiny_tokenizer()
    model = make_tiny_qwen2(len(tok), seed=seed, **model_kwargs)
    model.generation_config.pad_token_id = tok.pad_token_id
    model.generation_config.eos_token_id = tok.eos_token_id
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return path
