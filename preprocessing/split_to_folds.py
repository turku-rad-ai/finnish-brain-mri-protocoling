# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for splitting the training data into folds."
    )

    parser.add_argument("--folds", required=True, type=int)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--delimiter", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-csv", required=True)

    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"Incorrect CSV filename. {args.input_csv} does not exist.")
    else:
        data = pd.read_csv(args.input_csv, sep=args.delimiter, engine='python')
        data.index.name = 'orig_index'
        data['is_augmented'] = 0
        data['fold'] = ""
        
        print("Splitting the data into folds stratified by the protocol class.")
        kf = StratifiedKFold(n_splits=args.folds)
        for f, (_, test_index) in enumerate(kf.split(data['text'], data[args.target])):
            data.loc[test_index, 'fold'] = f
        
        data.to_csv(args.output_csv, sep=args.delimiter)
        print(f"Data is split into {args.folds} folds and saved.")
