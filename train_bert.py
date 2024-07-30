# -*- coding: utf-8 -*-
import argparse
import os
import pandas as pd
import time
import torch
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

# Define constants
TRAIN_BATCH = 64
VAL_BATCH = 32
MAX_LEN = 256
NUM_EPOCHS = 7

# Create The Dataset Class.
class TheDataset(torch.utils.data.Dataset):
	def __init__(self, texts, labels, tokenizer):
		self.texts      = texts
		self.labels     = labels
		self.tokenizer  = tokenizer
		self.max_len    = tokenizer.model_max_length
  
	def __len__(self):
		return len(self.texts)

	def __getitem__(self, index):
		text = str(self.texts[index])
		labels = self.labels[index]

		encoded_text = self.tokenizer.encode_plus(
			text,
			add_special_tokens    = True,
			max_length            = MAX_LEN,
			return_token_type_ids = False,
			return_attention_mask = True,
			return_tensors        = "pt",
			padding               = "max_length",
			truncation            = True
		)

		return {
			'input_ids': encoded_text['input_ids'][0],
			'attention_mask': encoded_text['attention_mask'][0],
			'labels': torch.tensor(labels, dtype=torch.long)
			}

# Custom function to compute metrics
def compute_metrics(pred):
	labels = pred.label_ids
	preds = pred.predictions.argmax(-1)

	f1_macro = f1_score(labels, preds, average='macro')
	f1_weighted = f1_score(labels, preds, average='weighted')
	acc = accuracy_score(labels, preds)

	return {
		'accuracy': acc,
		'f1_macro': f1_macro,
		'f1_weighted': f1_weighted
		}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to train the ML models."
    )

    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--delimiter", required=True)
    parser.add_argument("--bert-path", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    # Set the output folder and ensure that it is empty
    model_name = "bert_" + args.target
    model_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
		
    # Reading data
    data = pd.read_csv(args.train_csv, sep=args.delimiter, engine='python')
    num_labels = len(data[args.target].unique())

    # Load tokenizer for the BERT model
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)

    start_time = time.time()

    # k-fold cross validation
    for fold in range(int(args.folds)):
        start_fold_time = time.time()
        print("Fold ", fold)
        
        # Define the test and train sets for the fold
        train = data.loc[data['fold'] != fold]
        test = data.loc[(data['fold'] == fold) & (data['is_augmented'] != 1)]

        train_set_dataset = TheDataset(texts = train['text'].tolist(), labels = train[args.target].tolist(), tokenizer  = tokenizer)
        valid_set_dataset = TheDataset(texts = test['text'].tolist(), labels = test[args.target].tolist(), tokenizer = tokenizer)

        model = BertForSequenceClassification.from_pretrained(args.bert_path, num_labels = num_labels)

        # Create the fold where to save the trained model for this fold
        fold_output_dir = os.path.join(model_dir, 'fold{0}'.format(fold))
        os.makedirs(fold_output_dir, exist_ok=True)

        training_args = TrainingArguments(
                        output_dir                  = fold_output_dir,
                        num_train_epochs            = NUM_EPOCHS,
                        per_device_train_batch_size = TRAIN_BATCH,
                        per_device_eval_batch_size  = VAL_BATCH,
                        warmup_ratio                = 0.1,
                        learning_rate               = 5e-5,
                        weight_decay                = 0.01,
                        save_strategy               = "epoch",
                        evaluation_strategy         = "epoch",
                        logging_strategy            = "epoch",
                        save_total_limit            = 1,
                        load_best_model_at_end      = True
                )

        trainer = Trainer(
                        model           = model,
                        args            = training_args,
                        train_dataset   = train_set_dataset,
                        eval_dataset    = valid_set_dataset,
                        compute_metrics = compute_metrics
                )

        # Train the model and save
        trainer.train()
        trainer.save_model()

        # Save training logging history
        log_history = trainer.state.log_history
        with open(f"{fold_output_dir}/logging_history.txt", 'w') as f:
            for line in log_history:
                f.write(f"{line}\n")

        total_fold_time = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_fold_time))
        print("Training time for the fold: ", total_fold_time)

    total_time = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))
    print("Training completed. Total time for all folds: ", total_time)
