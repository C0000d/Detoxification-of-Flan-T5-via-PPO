# RLHF-Based Detoxification of Dialogue Summarization with Flan-T5

## Introduction  
In this project, I detoxified a **dialogue-summarization model** by fine-tuning **Flan-T5** with **PPO (Reinforcement Learning from Human Feedback)** so it learns to produce less toxic summaries.  
PPO fine-tuning involves four key components:  
- **Policy model** (trainable Flan-T5 + value head)  
- **Reference model** (frozen copy for KL regularization)  
- **Reward model** (RoBERTa hate-speech classifier)  
- **Value head / Critic** (estimates returns for PPO updates)  

---

## Process & Highlights  

1. **Dataset Preparation**  
   - Used [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum).  
   - Wrapped dialogues with instructions → built simple prompts.  
   - Tokenized into `query` and `input_ids`.  

2. **Base Model Setup**  
   - Loaded `google/flan-t5-base`.  
   - Attached a fine-tuned LoRA checkpoint.  
   - Wrapped with TRL’s `AutoModelForSeq2SeqLMWithValueHead` for PPO training.  
   - Cloned a frozen **reference model** to preserve correctness.  

3. **Reward Model**  
   - Used Meta AI’s **RoBERTa hate-speech classifier**.  
   - Extracted probability of `"nothate"` as the reward signal.  

4. **PPO Training**  
   - Used TRL’s `PPOTrainer`.  
   - Packed `(query, response, reward)` into PPO steps.  
   - Trained for **20 steps** with batch size = 16.  
   - Saved checkpoints for later evaluation.  

5. **Evaluation**  
   - Compared **reference model vs PPO-trained model**.  
   - Metric: toxicity mean & std (via Hugging Face `evaluate` toxicity module).  
   - Achieved **~40% reduction in both mean toxicity** and **variance** — a strong improvement for such a small training loop.  

---

## Challenges  

1. **Memory Bottlenecks on macOS (MPS backend)**  
   - PPO requires multiple models in memory → frequent crashes.  
   - Fix: split into three scripts:  
     - `data_preparation.py` (data loading & transformation)
     - `training_ppo.py` (training)  
     - `evaluate_ppo.py` (evaluation)  

2. **Reward Calculation Pitfall**  
   - Initially applied softmax → reward deltas shrank → model failed to learn.  
   - Fixed by using **raw logits** and **label** of `"nothate"` directly.  
   - This gave stable learning signals and consistent improvements.  

---

## Reflections  

- Gained **hands-on RLHF lifecycle experience**: dataset prep → LoRA fine-tuning → PPO training → evaluation.  
- Learned how to debug **memory bottlenecks** and maintain clean training/evaluation pipelines.  
- Discovered how reward signal design (logits vs softmax) critically affects PPO learning.  
- Overall, this was a **super valuable experience** in applying RLHF for safe language model fine-tuning.  

---

## Results  

Toxicity scores(the lower the better) before vs after PPO detoxification:  

| Metric      | Before | After  | Deduction |
|-------------|--------|--------|-------------|
| Mean        | 0.0336 | 0.0216 | **35.5%**  |
| Std         | 0.0447 | 0.0288 | **35.6%**  |

---

## Tech Stack  

- **Models:** Flan-T5, RoBERTa hate-speech classifier  
- **Libraries:** Hugging Face Transformers, TRL, PEFT (LoRA), Evaluate  
- **Training:** PPO fine-tuning with value head  
- **Hardware:** MacBook Pro M3 (MPS acceleration, low-resource setting)  

