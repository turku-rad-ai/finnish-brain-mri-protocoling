# -*- coding: utf-8 -*-
import argparse
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report
from preprocessing.preprocess_data import *

def evaluate_ml(data, protocol_model_path, contrast_model_path):
    protocol_model = pickle.load(open(protocol_model_path, 'rb')) 
    contrast_model = pickle.load(open(contrast_model_path, 'rb'))
    
    protocol_probs = protocol_model.predict_proba(data['text_preprocessed_for_ml'])
    data['protocol_pred_class'] = np.argmax(protocol_probs, axis=1)
    data['protocol_probs'] = protocol_probs.tolist()
    
    contrast_probs = protocol_model.predict_proba(data['text_preprocessed_for_ml'])
    data['contrast_pred_class'] = np.argmax(contrast_probs, axis=1)
    data['contrast_probs'] = contrast_probs.tolist()
    
    return data

def evaluate_bert(data, protocol_model_path, contrast_model_path):
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    protocol_model = BertForSequenceClassification.from_pretrained(protocol_model_path)
    contrast_model = BertForSequenceClassification.from_pretrained(contrast_model_path)
    
    encoded_input = tokenizer(data['text'].tolist(), return_tensors="pt", add_special_tokens=True)
    
    protocol_outputs = protocol_model(encoded_input["input_ids"], encoded_input["attention_mask"])
    protocol_probs = F.softmax(torch.tensor(protocol_outputs.logits.detach().numpy()), dim=1).numpy()
    protocol_pred_class = np.argmax(protocol_probs, 1)
    data['protocol_pred_class'] = protocol_pred_class
    data['protocol_probs'] = protocol_probs.tolist()

    contrast_outputs = contrast_model(encoded_input["input_ids"], encoded_input["attention_mask"])
    contrast_probs = F.softmax(torch.tensor(contrast_outputs.logits.detach().numpy()), dim=1).numpy()
    contrast_pred_class = np.argmax(contrast_probs, 1)
    data['contrast_pred_class'] = contrast_pred_class
    data['contrast_probs'] = contrast_probs.tolist()
        
    return data
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to evaluate a model."
    )

    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--delimiter", required=True)
    parser.add_argument("--spacy-path", required=True)
    parser.add_argument("--bert-path", required=True)
    parser.add_argument("--test-model", required=True)
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    print("------------------------------------------")
    print(f"EVALUATING THE MODEL: {args.test_model}")
    print("------------------------------------------")

    # Read the data
    data = pd.read_csv(args.test_csv, sep=args.delimiter, engine='python')
    nlp = spacy.load(args.spacy_path)
    
    # Do preprocessing and select evaluation function based on what model is tested
    if (args.test_model in ['nb', 'svc', 'xgb']):    
        protocol_model_path = f"{args.output_dir}/{args.test_model}_protocol/model.pkl"
        contrast_model_path = f"{args.output_dir}/{args.test_model}_contrast/model.pkl"
        data = do_preprocessing(data, nlp)
        evaluate = evaluate_ml
    if args.test_model == 'bert':
        protocol_model_path = f"{args.output_dir}/{args.test_model}_protocol/fold0/"
        contrast_model_path = f"{args.output_dir}/{args.test_model}_contrast/fold0/"
        data = do_preprocessing(data, nlp, do_preprocessing_for_ml=False)
        evaluate = evaluate_bert
    
    # Run evaluation and save the results in csv
    results = evaluate(data, protocol_model_path, contrast_model_path)
    results = results.rename(columns={"protocol": "protocol_true_class", "contrast": "contrast_true_class"})
    results_filepath = os.path.join(args.output_dir, (args.test_model + '_results.csv'))
    results.to_csv(results_filepath, sep=args.delimiter)
    
    print(results['protocol_pred_class'])
    print(results['contrast_pred_class'])
    
    print("------------------------------------------")
    print("RESULTS IN PREDICTING THE PROTOCOL CLASS:")
    print("------------------------------------------")
    print(classification_report(results['protocol_true_class'], results['protocol_pred_class'], zero_division=0))
    
    print("------------------------------------------")
    print("RESULTS IN PREDICTING THE CONTRAST CLASS:")
    print("------------------------------------------")
    print(classification_report(results['contrast_true_class'], results['contrast_pred_class'], zero_division=0))

