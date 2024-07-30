# -*- coding: utf-8 -*-

import argparse
import pandas as pd
from preprocessing.augmenter import Augmenter
from preprocessing.preprocess_data import clean_text

def augment_data(data, target, spacy_path):
  """Make data augmentations."""
  aug = Augmenter(spacy_path)
  augmented_data = []

  # If you want to augment some classes more than others, you can define here a list of classes and the augmentation factors (how much to augment each class)
  # Dictionary key is the class label (in numerical form) and the value is how many augmentations will be done for that class
  #augmentation_factor_list = {'0': 5, '1': 2, '2': 4, etc.}

  for i, row in data.iterrows():
    # how many augmentations to do for each referral if not using custom values for different classes
    augmentation_factor = 5

    # If you want to use the predefined augmentation factors for different classes, uncomment this. If a class is not defined in the list, the row will be augmented 5 times.
    #augmentation_factor = augmentation_factor_list.get(row[target], 5)

    for j in range(augmentation_factor):
      new_row = row.copy()
      new_row['text'] = aug.augment(row['text'])
      new_row['is_augmented'] = 1
      augmented_data.append(new_row)

  new_data = pd.concat([data, pd.DataFrame(augmented_data)], ignore_index=True)
  new_data = new_data.sort_values(by='orig_index')

  return new_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for augmenting the dataset."
    )

    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--delimiter", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--spacy-path", required=True)

    args = parser.parse_args()
    
    data = pd.read_csv(args.train_csv, sep=args.delimiter, engine='python')
    
    print('Cleaning the data.')
    data['text'] = data['text'].apply(clean_text)

    print('Starting to augment data.')
    data = augment_data(data, args.target, args.spacy_path)
    data.to_csv(args.train_csv, sep=args.delimiter)
    print('Augmentation is ready.')
