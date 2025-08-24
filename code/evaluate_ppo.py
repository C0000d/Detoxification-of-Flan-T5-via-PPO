import json, torch, evaluate
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
from peft import PeftModel
from datasets import load_from_disk
from trl import AutoModelForSeq2SeqLMWithValueHead
from tqdm import tqdm
import pandas as pd
import numpy as np

from data_preparation import (
    DATASET_DIR,
    MODEL_NAME,
)
from training_ppo import (
    CKPT_DIR,
    TOXICITY_MODEL_NAME,
    REF_MODEL_DIR,
    output_length_sampler,
    build_toxicity_pipeline,
)

NOT_HATE_INDEX = 0
# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def evaluate_toxicity(model,
                      toxicity_evaluator,
                      tokenizer,
                      dataset,
                      num_samples):
    """
    Evaluate the toxicity score on the dataset.
    :param toxicity_evaluator: the evaluator for the toxicity score
    :param tokenizer: tokenizer to be used
    :param dataset: dataset for evaluation
    :param num_samples: maximum number of samples to be processed in each run
    :return: (mean, std), where:
        mean: mean of all samples' toxicity score
        std: the standard deviation of the samples' toxicity score
    """
    max_new_tokens = 100
    toxicities = []
    for i, sample in tqdm(enumerate(dataset)):
        if i >= num_samples: break

        input_text = sample["query"]
        input_ids = tokenizer(input_text, return_tensors="pt", padding=True).input_ids
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            top_k=0.0,
            top_p=1,
            do_sample=True,
        )
        response = model.generate(input_ids=input_ids, generation_config=generation_config)
        response_text = tokenizer.decode(response[0], skip_special_tokens=True)

        toxicity_score = toxicity_evaluator.compute(predictions=[(input_text + " " + response_text)])
        toxicities.extend(toxicity_score["toxicity"])

    # compute the mean and std respectively
    mean = np.mean(toxicities)
    std = np.std(toxicities)
    return mean, std

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
# load test dataset
dataset = load_from_disk(DATASET_DIR)["test"]

# toxicity model
toxicity_evaluator = evaluate.load(
    "toxicity",
    TOXICITY_MODEL_NAME,
    module_type="measurement",
    toxic_label="hate"  # tell the evaluator the label of toxicity, since it has 2 cases: hate/nothate, offensive/not offensive
)

# load ref_model
ref_model = AutoModelForSeq2SeqLM.from_pretrained(str(REF_MODEL_DIR))

# load ppo checkpoint
ckpt_path = CKPT_DIR / "final_base"

tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path))
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
peft_model = PeftModel.from_pretrained(model, str(ckpt_path), torch_dtype=torch.bfloat16)
peft_model = peft_model.merge_and_unload()
ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
    peft_model,
    is_trainable=False,
    torch_dtype=torch.bfloat16,
).to("cpu").eval()

# benchmark evaluate the toxicity on test split
sentiment_pipe = build_toxicity_pipeline()
mean_before, std_before = evaluate_toxicity(model=ref_model,
                                          toxicity_evaluator=toxicity_evaluator,
                                          tokenizer=tokenizer,
                                          dataset=dataset,
                                          num_samples=10)
print(f"toxicity [mean, std] before detox: [{mean_before}, {std_before}]")

mean_after, std_after = evaluate_toxicity(model=ppo_model,
                                          toxicity_evaluator=toxicity_evaluator,
                                          tokenizer=tokenizer,
                                          dataset=dataset,
                                          num_samples=10)
print(f"toxicity [mean, std] after detox: [{mean_after}, {std_after}]")

mean_improvement = (mean_before - mean_after) / mean_before
std_improvement = (std_before - std_after) / std_before
print(f"Percentage improvement of toxicity score after detoxification:")
print(f"mean: {mean_improvement * 100:.2f}")
print(f"std: {std_improvement * 100:.2f}")

# Evaluate the model qualitively
# inspect some samples from the test split, compare the result against the ref model
batch_size = 20
compare_results = {}
reward_kwargs = {
    "top_k": None, # Return all scores.
    "function_to_apply": "none",
    "batch_size": 16
}

generation_config = {
    "min_length": 5,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
}
df_batch = dataset[0:batch_size]

compare_results["query"] = df_batch["query"]
prompt_tensors = df_batch["input_ids"]

summary_tensors_ref = []
summary_tensors = []

for i in tqdm(range(batch_size)):
    gen_len = output_length_sampler()
    generation_config["max_new_tokens"] = gen_len

    summary = ref_model.generate(
        input_ids=torch.as_tensor(prompt_tensors[i]).unsqueeze(dim=0),  # still process each prompt at a time, we have to add the dim of the batch_size to make the shape of the prompt [batch_size, seq_len]
        **generation_config
    ).squeeze()[-gen_len:]
    summary_tensors_ref.append(summary)

    summary = ppo_model.generate(
        input_ids=torch.as_tensor(prompt_tensors[i]).unsqueeze(dim=0),
        **generation_config
    ).squeeze()[-gen_len:]
    summary_tensors.append(summary)

# decode responses
compare_results["response_before"] = [tokenizer.decode(summary_tensors_ref[i]) for i in range(batch_size)]
compare_results["response_after"] = [tokenizer.decode(summary_tensors[i]) for i in range(batch_size)]

# Sentiment analysis of query/response pairs before/after
texts_before = [d + s for d, s in zip(compare_results["query"], compare_results["response_before"])]
rewards_before = sentiment_pipe(texts_before, **reward_kwargs)
compare_results["reward_before"] = [reward[NOT_HATE_INDEX]["score"] for reward in rewards_before]

texts_after = [d + s for d, s in zip(compare_results["query"], compare_results["response_after"])]
rewards_after = sentiment_pipe(texts_after, **reward_kwargs)
compare_results["reward_after"] = [reward[NOT_HATE_INDEX]["score"] for reward in rewards_after]

pd.set_option("display.max_colwidth", 500)
df_compare_results = pd.DataFrame(compare_results)
df_compare_results["reward_diff"] = df_compare_results["reward_after"] - df_compare_results["reward_before"]
df_compare_results_sorted = df_compare_results.sort_values(by=["reward_diff"], ascending=False).reset_index(drop=True)
print(df_compare_results_sorted.head(5))
