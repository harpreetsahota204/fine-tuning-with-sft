import argparse
import model_utils

def main():
    parser = argparse.ArgumentParser(description="Train a language model with custom configurations.")

    # Required arguments
    parser.add_argument("--model_id", type=str, required=True, help="The model identifier.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for training results.")
    parser.add_argument("--hf_username", type=str, required=True, help="Hugging Face username for pushing the model to the Hub.")
    parser.add_argument("--hf_repo_name", type=str, required=True, help="Repository name for the model on the Hugging Face Hub.")


    # Optional BnB configuration arguments
    parser.add_argument("--bnb_load_in_4bit", type=bool, default=True, help="Load model in 4-bit.")
    parser.add_argument("--bnb_4bit_use_double_quant", type=bool, default=True, help="Use double quantization for 4-bit.")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", help="Quantization type for 4-bit.")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="torch.bfloat16", help="Compute dtype for 4-bit.")

    # Optional LoRA configuration arguments
    parser.add_argument("--lora_r", type=int, default=16, help="Rank of LoRA layers.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha parameter for LoRA.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout rate for LoRA layers.")
    parser.add_argument("--lora_bias", type=str, default="none", help="Bias setting for LoRA.")
    parser.add_argument("--lora_target_modules", type=str, nargs='+', default=None, help="Target modules for LoRA.")

    # Optional training configuration arguments
    parser.add_argument("--evaluation_strategy", type=str, default="steps", help="Evaluation strategy.")
    parser.add_argument("--do_eval", type=bool, default=True, help="Whether to run evaluation.")
    parser.add_argument("--auto_find_batch_size", type=bool, default=True, help="Automatically find a suitable batch size.")
    parser.add_argument("--logging_steps", type=int, default=25, help="Number of steps for logging.")
    parser.add_argument("--save_steps", type=int, default=25, help="Number of steps after which to save the model.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--max_steps", type=int, default=125, help="Maximum number of training steps.")
    parser.add_argument("--warmup_steps", type=int, default=25, help="Number of warmup steps.")
    parser.add_argument("--bf16", type=bool, default=True, help="Use bf16 precision.")
    parser.add_argument("--tf32", type=bool, default=True, help="Use tf32 precision.")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True, help="Use gradient checkpointing.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--max_grad_norm", type=float, default=0.3, help="Maximum gradient norm.")
    parser.add_argument("--lr_scheduler_type", type=str, default="reduce_lr_on_plateau", help="Type of learning rate scheduler.")
    parser.add_argument("--report_to", type=str, default=None, help="Experiment tracking services to report to.")

    args = parser.parse_args()

    # Initialize model with BnB and LoRA configuration
    model, lora_config = model_utils.initialize_model(args.model_id, 
                                                      args.bnb_load_in_4bit, 
                                                      args.bnb_4bit_use_double_quant, 
                                                      args.bnb_4bit_quant_type, 
                                                      args.bnb_4bit_compute_dtype, 
                                                      args.lora_r, 
                                                      args.lora_alpha, 
                                                      args.lora_dropout, 
                                                      args.lora_bias, 
                                                      args.lora_target_modules
                                                      )

    # Initialize tokenizer
    tokenizer = model_utils.initialize_tokenizer(args.model_id)

    # Load dataset
    dataset = model_utils.load_dataset_custom(args.dataset)

    # Prepare training configuration
    training_args = model_utils.prepare_training_config(
        output_dir=args.output_dir,
        evaluation_strategy=args.evaluation_strategy,
        do_eval=args.do_eval,
        auto_find_batch_size=args.auto_find_batch_size,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        bf16=args.bf16,
        tf32=args.tf32,
        gradient_checkpointing=args.gradient_checkpointing,
        num_train_epochs=args.num_train_epochs,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type
    )

    # Train the model
    model_utils.train_model(model, 
                            tokenizer, 
                            dataset['train'], 
                            dataset['test'], 
                            training_args, 
                            lora_config
                            )
    
    merged_model = model_utils.merge_models(args.output_dir)
    
    model_utils.push_to_hub(merged_model, 
                            tokenizer, 
                            args.hf_username, 
                            args.hf_repo_name
                            )

if __name__ == "__main__":
    main()
