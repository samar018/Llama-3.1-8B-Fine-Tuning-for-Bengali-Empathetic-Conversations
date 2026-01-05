# Bengali Empathetic Chatbot

This repository contains a fine-tuned LLaMA 3.1 model for generating empathetic responses in Bengali. The model has been adapted using LoRA (Low-Rank Adaptation) on a curated Bengali empathetic dialogue dataset.

---

## 📚 Overview

The notebook demonstrates step-by-step:

1. **Loading the Pre-trained Model:**
   - Uses `meta-llama/Llama-3.1-8B-Instruct` as the base model.
   - Loads the model in 4-bit quantization using `bitsandbytes` for memory efficiency.

2. **Dataset Preparation:**
   - Contains curated Bengali sentences representing emotions like loneliness, sadness, anxiety, and more.
   - Each sample has an `input` (user text) and a `response` (expected empathetic reply).
   - Dataset is split into training (90%) and validation (10%).

3. **Formatting & Tokenization:**
   - Formats dataset in LLaMA instruct format with `<|user|>` and `<|assistant|>` tags.
   - Tokenizes using `transformers` tokenizer with proper EOS tokens and truncation.

4. **LoRA Fine-tuning:**
   - Configures LoRA adapters on specific linear layers of LLaMA 3.1.
   - Uses `Trainer` from Hugging Face with gradient accumulation, fp16, and evaluation steps.

5. **Training:**
   - Fine-tunes the model on the Bengali empathetic dataset.
   - Uses `DataCollatorForSeq2Seq` to handle dynamic padding.

6. **Evaluation:**
   - Evaluates the model using validation set.
   - Calculates perplexity to measure model performance.

7. **Generation:**
   - Generates empathetic responses to sample user inputs.
   - Uses LLaMA 3.1 specific prompt formatting for optimal instruction following.

8. **Testing:**
   - Includes multiple test cases for common user scenarios.
   - Generates output using the fine-tuned model and prints results in a readable format.

---

## 🛠 Files

- `Bengali_Empathetic_Chatbot.ipynb` - The main Jupyter notebook containing all steps.
- `README.md` - This file describing the project.

---

## 🔹 Test Cases Example

```python
test_cases = [
    "আমার খুব একা লাগছে",
    "আমি খুব হতাশ বোধ করছি",
    "কেউ আমাকে বোঝে না",
    "ভবিষ্যৎ নিয়ে আমি খুব চিন্তিত",
    "আমি মনে হয় জীবনে কিছুই করতে পারব না",
]

for text in test_cases:
    output = test_finetuned_model(model, tokenizer, text)
    print("User:", text)
    print("Assistant:", output)
    print("-"*60)
```

---

## ⚙️ Requirements

- Python 3.10+
- `transformers`
- `accelerate`
- `datasets`
- `peft`
- `bitsandbytes`
- `torch`

Install dependencies via:

```bash
pip install transformers accelerate peft datasets evaluate bitsandbytes
```

---

## 📌 Notes

- Ensure GPU support for efficient fine-tuning.
- Use HF token for authentication when downloading the model.
- Fine-tuning hyperparameters like `num_train_epochs` and `gradient_accumulation_steps` can be adjusted based on your GPU memory.

---

## 🔗 Kaggle Notebook

[Open Notebook in Kaggle](https://www.kaggle.com/code/samardas/fine-tuning-llama-bengali-task/edit)

