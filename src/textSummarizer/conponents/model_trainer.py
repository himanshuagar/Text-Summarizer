from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
import torch
from textSummarizer.entity import ModelTrainerConfig
import os

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        # Device detection with clear error handling
        if torch.backends.mps.is_available():
            device = "mps"
            print("🚀 Using Apple Silicon GPU (MPS backend)")
        elif torch.cuda.is_available():
            device = "cuda"
            print("🚀 Using NVIDIA GPU (CUDA backend)")
        else:
            device = "cpu"
            print("⚠️ Using CPU - training will be slow")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        
        # Truncation settings to reduce memory usage
        max_input_length = 512
        max_output_length = 128
        
        # Load and process dataset
        try:
            dataset_samsum_pt = load_from_disk(self.config.data_path)
            print(f"✅ Loaded dataset from {self.config.data_path}")
        except Exception as e:
            print(f"❌ Failed to load dataset: {str(e)}")
            raise
        
        def truncate_example(example):
            return {
                'input_ids': example['input_ids'][:max_input_length],
                'attention_mask': example['attention_mask'][:max_input_length],
                'labels': example['labels'][:max_output_length]
            }
        
        # Select smaller subsets for training and validation
        train_dataset = dataset_samsum_pt["train"].select(range(50)).map(truncate_example)
        eval_dataset = dataset_samsum_pt["validation"].select(range(20)).map(truncate_example)
        print(f"📊 Training on {len(train_dataset)} samples, validating on {len(eval_dataset)} samples")

        # Use dynamic padding to optimize memory usage
        seq2seq_data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model_pegasus,
            padding='longest'
        )

        # Configure TrainingArguments - disable all mixed precision for MPS
        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=self.config.num_train_epochs,
            warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=1,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            gradient_checkpointing=False,  # Disabled for MPS compatibility
            fp16=False,  # Disabled for MPS
            bf16=False,  # Disabled for MPS
            optim="adamw_torch",  # Recommended optimizer
            report_to="none",     # Disable external logging
            no_cuda=(device != "cuda")  # Explicitly disable CUDA if not using it
        )

        print("⚙️ Training arguments configured:")
        print(f"• Device: {device}")
        print(f"• Batch size: {self.config.per_device_train_batch_size}")
        print(f"• Epochs: {self.config.num_train_epochs}")
        print(f"• Precision: full (fp32)")

        trainer = Trainer(
            model=model_pegasus,
            args=trainer_args,
            tokenizer=tokenizer,
            data_collator=seq2seq_data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        print("🏁 Starting training...")
        try:
            trainer.train()
            print("🎉 Training completed successfully!")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("💥 Out of memory error! Try these solutions:")
                print("1. Reduce max_input_length (current: 512)")
                print("2. Reduce batch size (current: {self.config.per_device_train_batch_size})")
                print("3. Reduce dataset subset sizes (current: train 50, eval 20)")
                print("4. Use gradient checkpointing by setting gradient_checkpointing=True")
            raise
        
        # Save trained model and tokenizer
        output_dir = os.path.join(self.config.root_dir, "pegasus-samsum-model")
        tokenizer_dir = os.path.join(self.config.root_dir, "tokenizer")
        try:
            model_pegasus.save_pretrained(output_dir)
            tokenizer.save_pretrained(tokenizer_dir)
            print(f"💾 Model saved to {output_dir}")
        except Exception as e:
            print(f"❌ Failed to save model: {str(e)}")
            raise
