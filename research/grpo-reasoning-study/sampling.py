"""
Two interchangeable ways to sample completions: Hugging Face `generate()` (simple, runs anywhere,
slow) and vLLM (paged attention + continuous batching, ~10x faster sampling).

In the earlier TRL run, HF sampling took most of the 12-15 s step time, so the sweep in this
study uses vLLM by default. The policy is trained in PyTorch and vLLM keeps its own copy of the weights,
so after every optimizer step the new weights are copied into the vLLM engine (`sync`).

Both samplers draw from the *unmodified* model distribution: temperature only, no top-k, no top-p,
no repetition penalty. Qwen2.5-Instruct's generation_config defaults to top_k=20, top_p=0.8 and a 1.05
repetition penalty. Sampling with those settings would make the training data off-policy, while
GRPO's ratio pi_theta / pi_old assumes the samples came from pi_old itself.
"""

import os
from dataclasses import dataclass

import torch


@dataclass
class Completion:
    token_ids: list  # generated tokens, including the EOS token when the model stopped by itself
    text: str
    truncated: bool  # hit max_new_tokens without emitting EOS


class HFSampler:
    """Samples with the training model itself (no weight copy needed)."""

    def __init__(self, model, tokenizer, max_new_tokens, batch_size=16):
        self.model, self.tokenizer = model, tokenizer
        self.max_new_tokens, self.batch_size = max_new_tokens, batch_size

    @torch.no_grad()
    def generate(self, prompt_ids, n, temperature=1.0, greedy=False, seed=None):
        """prompt_ids: list of token-id lists -> list (per prompt) of n Completions."""
        if seed is not None:
            torch.manual_seed(seed)
        model, tok = self.model, self.tokenizer
        # Generation with gradient checkpointing on and the model in train mode silently drops the KV
        # cache (a bug hit in mini-chatgpt's DPO run), so switch both off while sampling.
        was_training, was_ckpt = model.training, getattr(model, "is_gradient_checkpointing", False)
        model.eval()
        if was_ckpt:
            model.gradient_checkpointing_disable()
        results = []
        try:
            device = next(model.parameters()).device
            for start in range(0, len(prompt_ids), self.batch_size):
                chunk = prompt_ids[start : start + self.batch_size]
                width = max(len(p) for p in chunk)
                ids = torch.full((len(chunk), width), tok.pad_token_id, dtype=torch.long)
                mask = torch.zeros_like(ids)
                for i, p in enumerate(chunk):  # left padding: generation continues from the right
                    ids[i, width - len(p) :] = torch.tensor(p)
                    mask[i, width - len(p) :] = 1
                sampling = (
                    {"do_sample": False}
                    if greedy
                    else {"do_sample": True, "temperature": temperature, "top_k": 0, "top_p": 1.0}
                )
                out = model.generate(
                    input_ids=ids.to(device),
                    attention_mask=mask.to(device),
                    max_new_tokens=self.max_new_tokens,
                    num_return_sequences=n,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                    repetition_penalty=1.0,
                    use_cache=True,
                    **sampling,
                )
                new_tokens = out[:, width:].tolist()
                for i in range(len(chunk)):
                    results.append([self._to_completion(t) for t in new_tokens[i * n : (i + 1) * n]])
        finally:
            if was_ckpt:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train(was_training)
        return results

    def _to_completion(self, tokens):
        eos = self.tokenizer.eos_token_id
        if eos in tokens:
            tokens = tokens[: tokens.index(eos) + 1]
            truncated = False
        else:
            truncated = True
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return Completion(tokens, text, truncated)

    def sync(self, model):
        pass  # same weights


class VLLMSampler:
    """vLLM engine colocated with training on the same GPU.

    Weight sync uses vLLM internals, which have changed between releases. `sync` tries the
    in-process paths of the V1 and V0 engines. If neither exists, it falls back to the public
    `LLM.apply_model`, which is slower because the weights are pickled to the engine process. The
    vLLM path was written against vLLM 0.10/0.11 and was not run in the CPU-only environment where
    this code was written. Check `results/runs/*/train_log.jsonl`: if the weight sync fails silently,
    the correctness reward will stay flat.
    """

    def __init__(self, model_path, tokenizer, max_new_tokens, gpu_memory_utilization=0.3, max_model_len=2048, seed=0):
        # Keep the engine in this process so its model object is reachable for weight updates.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from vllm import LLM, SamplingParams

        self._SamplingParams = SamplingParams
        self.tokenizer, self.max_new_tokens = tokenizer, max_new_tokens
        self.llm = LLM(
            model=model_path,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            seed=seed,
            enable_prefix_caching=True,
        )

    def generate(self, prompt_ids, n, temperature=1.0, greedy=False, seed=None):
        params = self._SamplingParams(
            n=n,
            temperature=0.0 if greedy else temperature,
            top_p=1.0,
            repetition_penalty=1.0,
            max_tokens=self.max_new_tokens,
            seed=seed,
        )
        outputs = self.llm.generate([{"prompt_token_ids": p} for p in prompt_ids], params, use_tqdm=False)
        eos = self.tokenizer.eos_token_id
        results = []
        for out in outputs:
            group = []
            for o in out.outputs:
                tokens = list(o.token_ids)
                stopped = o.finish_reason == "stop"
                if stopped and (not tokens or tokens[-1] != eos):
                    tokens.append(eos)  # train on the stop token so the model learns to end
                group.append(Completion(tokens, o.text, truncated=not stopped))
            results.append(group)
        return results

    def _engine_model(self):
        engine = self.llm.llm_engine
        for path in (
            "model_executor.driver_worker.worker.model_runner.model",  # V1, in-process
            "model_executor.driver_worker.model_runner.model",
        ):  # V0
            obj = engine
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                return obj
            except AttributeError:
                continue
        return None

    @torch.no_grad()
    def sync(self, model):
        weights = [(name, p.detach()) for name, p in model.named_parameters()]
        target = self._engine_model()
        if target is not None:
            target.load_weights(weights)
        else:
            self.llm.apply_model(lambda m: m.load_weights(weights))
        self.llm.reset_prefix_cache()  # cached KV blocks were computed with the old weights
