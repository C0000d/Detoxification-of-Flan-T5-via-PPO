from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

MODEL_NAME = "google/flan-t5-base"
DATASET_NAME = "knkarthick/dialogsum"
DATASET_DIR = Path("data")
# Must-haves for PPOTrainer: Query, Response, Reward
# preprocess the dataset
# wrap them with the instruction then tokenize
# prepare the input_ids and query attributes
def build_dataset(model_name, dataset_name, input_min_text_length, input_max_text_length):
    # load dataset, keep the train part only
    dataset = load_dataset(dataset_name, split="train")

    # filter the examples that have appropriately long dialogues
    dataset = dataset.filter(lambda x: len(x["dialogue"]) > input_min_text_length and len(x["dialogue"]) < input_max_text_length)

    # load tokenizer, set the device_map to auto to allow the computation switch automatically between GPU and CPU
    tokenizer = AutoTokenizer.from_pretrained(model_name) # the device_map attribute will only affect when we load the model

    def tokenize(sample):
        prompt = f"""
Summarize the following conversation:

{sample["dialogue"]}

Summary:
"""
        sample["input_ids"] = tokenizer.encode(prompt)

        # prepare the query for PPOTrainer
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    # tokenize each dialogue
    dataset = dataset.map(tokenize, batched=False) # apply one example at a time
    dataset.set_format(type="torch")

    # split the dataset into train and test
    dataset_splits = dataset.train_test_split(test_size=0.2, shuffle=False, seed=42)

    return dataset_splits

def main():
    dataset = build_dataset(model_name=MODEL_NAME,
                            dataset_name=DATASET_NAME,
                            input_min_text_length=200,
                            input_max_text_length=1000)

    dataset.save_to_disk(DATASET_DIR)
    print(f"Saved tokenized dataset <UNK> {DATASET_DIR}")

if __name__ == "__main__":
    main()