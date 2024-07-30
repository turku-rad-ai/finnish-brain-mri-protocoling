# -*- coding: utf-8 -*-

import argparse
import re
import pandas as pd
import spacy

def clean_text(string):
  """Remove extra whitespaces and do other preprocessing if needed."""
  string = " ".join(string.split())

  # Our RIS system adds some unnecessary sentences (which end in ': : ') in the beginning of the referrals, this is how we removed them.
  #split_str = [": : "]
  #if any(x in string for x in split_str):
  #  string = re.split(r'(?:' + '|'.join(split_str) + r')', string)[1]
  #else:
  #  string = string

  return string

def preprocessor_for_ml(text, nlp):
  """Preprocess the data for ML models: lemmatize, remove stopwords and punctuations"""
  lemmatized = ' '.join([token.lemma_ for token in nlp(text)])
  cleaned = [token for token in nlp(lemmatized) if not token.is_stop and not token.is_punct]
  text = ' '.join([str(token) for token in cleaned])

  return(text)
  
def do_preprocessing(data, nlp, do_preprocessing_for_ml=True):
    data['text'] = data['text'].apply(clean_text)
    # Lemmatization can be skipped in specific cases, e.g. when evaluating the BERT model
    if do_preprocessing_for_ml:
        data['text_preprocessed_for_ml'] =  data['text'].apply(preprocessor_for_ml, nlp=nlp)
    
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for cleaning and preprocessing the data for ML models."
    )

    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--delimiter", required=True)
    parser.add_argument("--spacy-path", required=True)

    args = parser.parse_args()

    nlp = spacy.load(args.spacy_path)

    print('Starting to clean and preprocess the data.')

    data = pd.read_csv(args.train_csv, sep=args.delimiter, engine='python')
    data = do_preprocessing(data, nlp)
    data.to_csv(args.train_csv, sep=args.delimiter)


    print('Preprocessing is ready.')
