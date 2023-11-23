### 🚀 Fine-Tuning with SFT (Supervised Fine Tuning)

Welcome to `fine-tuning-with-sft`, a repository by [harpreetsahota204](https://github.com/harpreetsahota204). This project facilitates supervised fine-tuning of large language models, using advanced techniques like LoRA and BitsAndBytes for efficient and effective training.

---

### 📋 Description

I created this repo to help keep my fine-tuning notebooks clean. It allows for customization through command-line arguments to meet diverse training needs.

---

### 📥 Installation

Clone and set up the project with ease:

```bash
git clone https://github.com/harpreetsahota204/fine-tuning-with-sft.git
cd fine-tuning-with-sft
pip install -r requirements.txt
```

🚨 Special Note: `flash-attn` installation:

```bash
pip install flash-attn --no-build-isolation
```

---

### 🚀 Usage

Jumpstart your model training:

```bash
python train_model.py --model_id "YourModel" --dataset "YourDataset" --output_dir "./output"
```

Explore various command-line arguments in `train_model.py` for tailor-made training setups.

---

### 🛠️ Configuration

#### 🎛️ LoRA Configurations
- `--lora_r`: Rank of LoRA layers (default: 16).
- `--lora_alpha`: Alpha parameter for LoRA (default: 32).
- `--lora_dropout`: Dropout rate for LoRA layers (default: 0.1).
- `--lora_bias`: Bias setting for LoRA (default: "none").
- `--lora_target_modules`: Target modules for LoRA (default: ['gate_proj', 'down_proj', 'up_proj']).

#### 🔢 BnB Configurations
- `--bnb_load_in_4bit`: Load model in 4-bit (default: True).
- `--bnb_4bit_use_double_quant`: Use double quantization for 4-bit (default: True).
- `--bnb_4bit_quant_type`: Quantization type for 4-bit (default: "nf4").
- `--bnb_4bit_compute_dtype`: Compute dtype for 4-bit (default: "torch.bfloat16").

#### 🏋️‍♂️ Training Parameters
- `--evaluation_strategy`: Evaluation strategy (default: "steps").
- `--do_eval`: Whether to run evaluation (default: True).
- `--auto_find_batch_size`: Automatically find a suitable batch size (default: True).
- `--logging_steps`: Number of steps for logging (default: 25).
- `--save_steps`: Number of steps after which to save the model (default: 25).
- `--learning_rate`: Learning rate (default: 3e-4).
- `--weight_decay`: Weight decay (default: 0.01).
- `--max_steps`: Maximum number of training steps (default: 125).
- `--warmup_steps`: Number of warmup steps (default: 25).
- `--bf16`: Use bf16 precision (default: True).
- `--tf32`: Use tf32 precision (default: True).
- `--gradient_checkpointing`: Use gradient checkpointing (default: True).
- `--num_train_epochs`: Number of training epochs (default: 1).
- `--max_grad_norm`: Maximum gradient norm (default: 0.3).
- `--lr_scheduler_type`: Type of learning rate scheduler (default: "reduce_lr_on_plateau").


---

### 🤖 Model and Tokenizer

Powered by Hugging Face's `transformers` and optimized with LoRA and BnB for peak performance.

---

### 📈 Training

Managed by the `SFTTrainer` from `trl`, custom-tailored for our configurations.

---

### 💡 Example Command

Kick off training with:

```bash
python train_model.py --model_id "Deci/DeciLM-6b" --dataset "TeamDLD/neurips_challenge_dataset" --output_dir "./output" --lora_r 16 --lora_alpha 32
```

---

### 📚 Documentation

Dive deeper into each function and class in `train_model.py` and `model_utils.py`.

---

### ✏️ Contributing

This repository was quickly created to aid in specific tasks. Any and all contributions to enhance its robustness are warmly welcomed!

---

### 📄 License

This project is open-sourced under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0), aligning with the default Hugging Face license.

---

### 📞 Contact

Got questions or feedback? Reach out to [@datascienceharp on Twitter](https://twitter.com/datascienceharp) or join the [Deep Learning Daily Discord community](https://www.deeplearningdaily.community/).

[![Twitter Follow](https://img.shields.io/twitter/follow/datascienceharp?style=social)](https://twitter.com/datascienceharp)
[![Discord](https://img.shields.io/discord/1081284435405717566?label=join%20discord&logo=discord&style=social)](https://www.deeplearningdaily.community/)

---

### 🙏 Acknowledgments

- Kudos to Hugging Face for the transformative `transformers` and `datasets` libraries.
- Gratitude to the TRL and peft teams for their innovative training methodologies.

