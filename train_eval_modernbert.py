import argparse
import os.path
import random
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, DataCollatorWithPadding, Trainer, logging, set_seed


def train_eval(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataset = load_dataset("csv", data_files=args.train_data)

    def tokenize_function(example):
        return tokenizer(example[args.train_text_column], truncation=True)

    dataset = dataset.rename_column(args.label_column, "labels")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=[args.train_text_column])

    print(tokenized_dataset)

    if not os.path.exists(args.model_save_dir):
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=1)

        training_args = TrainingArguments(
            output_dir=args.model_save_dir,
            do_eval=False,
            eval_strategy='no',
            save_strategy='no',
            learning_rate=5e-5,
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=5,
            weight_decay=0.01,
            bf16=True,
            gradient_accumulation_steps=args.gcs,
            logging_strategy='steps',
            logging_steps=100,
            optim="adamw_torch_fused",
            report_to="none",  # Disable logging to wandb etc.
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            data_collator=data_collator,
        )

        trainer.train()
        trainer.save_model(args.model_save_dir)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_save_dir, num_labels=1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    model.eval()
    for path in args.eval_data:
        eval_df = pd.read_csv(path)
        eval_texts = eval_df[args.eval_text_column].tolist()
        eval_data = tokenizer(eval_texts, padding=True, truncation=True, return_tensors="pt")
        test_dataset = torch.utils.data.TensorDataset(eval_data['input_ids'], eval_data['attention_mask'])
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask = [b.to(model.device) for b in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.squeeze(-1).cpu().numpy()
                predictions.extend(preds)

        eval_df['prediction'] = predictions
        eval_df.to_csv(path.replace('.csv', f'{args.output_name}.csv'), index=False, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="answerdotai/ModernBERT-large")
    parser.add_argument("--model_save_dir", type=str, required=True)
    parser.add_argument("--train_text_column", type=str, default="text")
    parser.add_argument("--eval_text_column", type=str, default="text")
    parser.add_argument("--label_column", type=str, default="labels")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument('--eval_data', nargs='+', help='List of eval data', required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gcs", type=int, default=1)
    parser.add_argument("--output_name", type=str, default='')
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)

    logging.set_verbosity_info()

    train_eval(args)
