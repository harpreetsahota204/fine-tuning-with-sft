from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from datasets import load_dataset, load_from_disk
import torch
import os

# Initialize the model with BnB and LoRA configuration
def initialize_model(model_id, bnb_load_in_4bit, bnb_4bit_use_double_quant, bnb_4bit_quant_type, bnb_4bit_compute_dtype, lora_r, lora_alpha, lora_dropout, lora_bias, lora_target_modules):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=bnb_load_in_4bit,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=eval(bnb_4bit_compute_dtype)
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        use_cache=False,
        trust_remote_code=True,
        use_flash_attention_2=True
    )

    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        target_modules=lora_target_modules,
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    return model, lora_config

# Initialize the tokenizer
def initialize_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

# Load dataset
def load_dataset_custom(dataset_name):
    return load_from_disk(dataset_name)

# Prepare training configuration
def prepare_training_config(output_dir, **kwargs):
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy=kwargs.get("evaluation_strategy", "steps"),
        do_eval=kwargs.get("do_eval", True),
        auto_find_batch_size=kwargs.get("auto_find_batch_size", True),
        logging_steps=kwargs.get("logging_steps", 25),
        save_steps=kwargs.get("save_steps", 25),
        learning_rate=kwargs.get("learning_rate", 3e-4),
        weight_decay=kwargs.get("weight_decay", 0.01),
        max_steps=kwargs.get("max_steps", 125),
        warmup_steps=kwargs.get("warmup_steps", 25),
        bf16=kwargs.get("bf16", True),
        tf32=kwargs.get("tf32", True),
        gradient_checkpointing=kwargs.get("gradient_checkpointing", True),
        num_train_epochs=kwargs.get("num_train_epochs", 1),
        max_grad_norm=kwargs.get("max_grad_norm", 0.3),
        lr_scheduler_type=kwargs.get("lr_scheduler_type", "reduce_lr_on_plateau"),
        report_to=kwargs.get("report_to", None)
    )
    return training_args

# Train the model
def train_model(model, tokenizer, train_dataset, val_dataset, training_args, lora_config):
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=lora_config,
        tokenizer=tokenizer,
        dataset_text_field='text',
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        max_seq_length=4096,
        dataset_num_proc=os.cpu_count(),
        packing=True
    )
    trainer.train()
    trainer.save_model()

def merge_models(output_dir):
    # Load the trained model for merging
    tuned_model = AutoPeftModelForCausalLM.from_pretrained(
        output_dir,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )

    # Perform the merge operation
    merged_model = tuned_model.merge_and_unload()
    
    return merged_model

def push_to_hub(model, tokenizer, hf_username, hf_repo_name):
    # Push the merged model to the Hub
    model.push_to_hub(f"{hf_username}/{hf_repo_name}", private=True)
    
    # Push the tokenizer to the Hub
    tokenizer.push_to_hub(f"{hf_username}/{hf_repo_name}", private=True)
