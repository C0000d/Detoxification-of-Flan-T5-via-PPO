# Fine-tuning the Flan-T5 model with PPO and PEFT
# with Meta AI's hate speech reward model
from functools import lru_cache

from torch._C import device
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, GenerationConfig
from datasets import load_dataset, load_from_disk
from peft import PeftModel, PeftConfig, LoraConfig, TaskType

# trl: Transformer Reinforcement Learning library
from trl.trainer import (PPOTrainer, PPOConfig)
from trl import AutoModelForSeq2SeqLMWithValueHead # a custom wrapper designed for RLHF tasks
from trl import create_reference_model
from trl.core import LengthSampler

import torch
import evaluate

import numpy as np
import pandas as pd

# show the progress bar for loops
from tqdm import tqdm
tqdm.pandas()

from pathlib import Path
import json, datetime as dt
from data_preparation import (
    DATASET_DIR,
    MODEL_NAME,
)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
TOXICITY_MODEL_NAME = "facebook/roberta-hate-speech-dynabench-r4-target"
NOT_HATE_INDEX = 0  # index 0 out of [not hate, hate], to get the logits for the not hate
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)
REF_MODEL_DIR = Path("checkpoints/ref_model")
REF_MODEL_DIR.mkdir(exist_ok=True)
GEN_CFG = {
    "max_new_tokens": 200,
    "top_p": 0.0,
    "top_k": 1,
    "do_sample": True,
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0

    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return (f"\ntrainable model parameters: {trainable_model_params}\n\
all model parameters: {all_model_params}\n\
percentage of trainable model parameters: {100 * trainable_model_params / all_model_params: .2f}%")

def collator(data):
    """
    Convert a list of N dicts into a dict of lists (length N).
    :param data: [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    :return: {'a': [1, 3], 'b': [2, 4]}
    """
    return dict((key, [d[key] for d in data]) for key in data[0])

@lru_cache(maxsize=1)
def build_toxicity_pipeline():
    # a binary classifier which will output the score for both [not hate, hate]
    toxicity_tokenizer = AutoTokenizer.from_pretrained(TOXICITY_MODEL_NAME)
    toxicity_model = AutoModelForSequenceClassification.from_pretrained(
        TOXICITY_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        # device_map="auto"
    )
    return pipeline(
        "sentiment-analysis",
        model=toxicity_model,
        tokenizer=toxicity_tokenizer,
        device_map="auto",
        function_to_apply="softmax",
        top_k=None,
        batch_size=16,
    )

def save_ckpt(tag: str, ppo_model, tokenizer, stats=None):
    """Save LoRA & value head + run metadata."""
    tag += "_base"
    out_dir = CKPT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Trainable weights (LoRA + value head)
    ppo_model.save_pretrained(out_dir)

    # 2. Tokenizer (once is fine; overwrite = cheap)
    tokenizer.save_pretrained(out_dir)

    # 3. Run metadata — whatever you find useful
    if stats:
        meta = {
            "step": tag,
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "objective_kl": float(stats["objective/kl"]),
            "return_mean": float(stats["ppo/returns/mean"]),
            "advantage_mean": float(stats["ppo/policy/advantages_mean"]),
        }
        with open(out_dir / "run_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    print(f"Saved checkpoint → {out_dir}")

# ---------------------------------------------------------------------------
# 1. Model & Data Preparations
# ---------------------------------------------------------------------------
# 1.a. Load dataset
dataset = load_from_disk(DATASET_DIR)["train"]

# 1.b. Prepare the PPO model
# we load a fine-tuned peft checkpoint(peft adapter's weight) from AWS and fit it to the base model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME,
                                              torch_dtype=torch.bfloat16,
                                              )

lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
)

peft_model = PeftModel.from_pretrained(model, # base model
                                       "./peft-dialogue-summary-checkpoint-from-s3", # the checkpoint we loaded
                                       lora_config=lora_config,
                                       torch_dtype=torch.bfloat16, # cuts memory use in half compares to default(float32)
                                       is_trainable=True)
"""
trainable model parameters: 3538944
all model parameters: 251116800
percentage of trainable model parameters:  1.41%
"""

# then we attach a value head to the peft model to make it a ppo model(where the value head is used as the critic model)
ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
    peft_model,
    torch_dtype=torch.bfloat16,
    is_trainable=True,
)
"""
the output structure — deliberately tiny, to keep the hit on VRAM/latency minimal
ValueHead(
  (dropout): Dropout(p=0.1, inplace=False) — randomly zeros 10% of hidden units druing training, to regularize the value head so it doesn't overfit to a small PPO batch.
  (summary): Linear(in_features=768, out_features=1, bias=True) — B * L * 768 -> B * L * 1: convert each token's last-layer hidden state into a scalar value estimate Vt
  (flatten): Flatten(start_dim=1, end_dim=-1) — B * L * 1 -> B * L: downstream the tensor
)
where B = batch_size, L = sequence_length, 768 = the embedding width of FLAN-T5-Base
"""

# create a frozen copy of the PPO to be the reference model.
ref_model = create_reference_model(ppo_model)
ref_model.save_pretrained(REF_MODEL_DIR)
"""
trainable model parameters: 0
all model parameters: 251117569
percentage of trainable model parameters:  0.00%
"""

# 1.c. Prepare the reward model
# we'll directly use the Meta's RoBERTa-based hate speech model here
# it will provide the score by 2 labels: nothate(0), hate(1)
sentiment_pipe = build_toxicity_pipeline() # an inference tool — you pass it a list of texts, it'll run the model

# set up the toxicity evaluation metric
# toxicity score range: 0~1, larger the number, higher the toxicity
toxicity_evaluator = evaluate.load("toxicity",
                                   TOXICITY_MODEL_NAME,
                                   module_type="measurement",  # one number per input
                                   toxic_label="hate") # tell the toxicity module the label name inside the model corresponds to toxic

# ---------------------------------------------------------------------------
# 3. Perform Detoxification Fine-Tuning
# ---------------------------------------------------------------------------
# 3.a Initialize the PPOTrainer
learning_rate=1.41e-5  # the optimiser lr for ppo (LoRA weight & value head)
max_ppo_epochs=1    # the number of epoch we train on the same batch of data
mini_batch_size=4   # each batch will be split into 4 mini batches in optimiser in each PPO iteration (so here we have 16/4 = 4 gradient steps per PPO epoch)
batch_size=16   # collect 16 query-response pair and score them in one PPO iteration

config=PPOConfig(
    model_name=MODEL_NAME,
    learning_rate=learning_rate,
    ppo_epochs=max_ppo_epochs,
    mini_batch_size=mini_batch_size,
    batch_size=batch_size,
    init_kl_coef=0.1
) # tells the trainer how to optimise once it has (prompt, response, reward) tensors

ppo_trainer = PPOTrainer(
    config=config,
    model=ppo_model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator  # so the dataset becomes {"input_ids: [...], ..."} after the dataloader
)

# 3.b Fine-tune the model
# sampling prompts, generating responses, scoring with the reward model, getting the response & reward, feed the (query, response, reward) into the ppo trainer.
output_min_length = 100
output_max_length = 400
output_length_sampler = LengthSampler(output_min_length, output_max_length)  # randomly pick the max_new_tokens within [100, 400]

generation_kwargs = {
    "min_length": 5,
    "top_k": 0.0,
    "top_p": 1,
    "do_sample": False,
}

max_ppo_steps = 20
SAVE_EVERY    = 5   # save every 5 step
def main():
    for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):  # built from the training split and the collator
        if step > max_ppo_steps:
            break

        prompt_tensors = batch["input_ids"]

        # get response from the model
        summary_tensors = []
        for prompt_tensor in prompt_tensors:
            max_new_tokens = output_length_sampler()  # we use different max_new_tokens for each sample, exposes the policy model to variable length generations

            GEN_CFG["max_new_tokens"] = max_new_tokens
            summary = ppo_trainer.generate(prompt_tensor, **GEN_CFG).squeeze() # remove the dim produced by the generate() since the batch_size is 1 as we input one prompt per loop, keep the response's tokens only
            summary_tensors.append(summary[-max_new_tokens:])  # discard the prompt and keep exactly the response from the output

        # pack the responses into the batch
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in summary_tensors]

        # compute reward outputs
        query_response_pairs = [q+r for q, r in zip(batch["query"], batch["response"])]
        rewards = sentiment_pipe(query_response_pairs)
        p_hate_list = [1.0 - next(d["score"] for d in reward if d["label"] == "nothate") for reward in rewards]
        reward_tensors = [torch.tensor(p) for p in p_hate_list] # reward: {"label": "nothate"/"hate", "score": ...}

        # reward
        stats = ppo_trainer.step(prompt_tensors, summary_tensors, reward_tensors)
        ppo_trainer.log_stats(stats, batch, reward_tensors)

        print(f"step {step+1:03d} | kl {stats['objective/kl']:.2f} | "
              f"return {stats['ppo/returns/mean']:.3f} | advantages_mean: {stats["ppo/policy/advantages_mean"]}")

        # save checkpoints
        if (step + 1) % SAVE_EVERY == 0:
            save_ckpt(f"step_{step + 1:06d}", ppo_model, tokenizer, stats)

    # final checkpoint
    save_ckpt("final", ppo_model, tokenizer)

if __name__ == "__main__":
    main()