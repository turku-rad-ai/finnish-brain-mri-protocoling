# -*- coding: utf-8 -*-
import argparse
import os
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import SVC
import xgboost as xgboost


class CustomSplit(PredefinedSplit):
  """Custom PreDefinedSplit class which uses predefined 5-fold split and excludes augmented samples from the validation set."""
  def __init__(self, test_fold, is_augmented):
    super().__init__(test_fold)
    self.is_augmented = is_augmented[is_augmented == 1].index.tolist()
  
  def split(self, X=None, y=None, groups=None):
        ind = np.arange(len(self.test_fold))
        for test_index in self._iter_test_masks():
            train_index = ind[np.logical_not(test_index)]
            test_index = ind[test_index]    
            # Delete augmented data from test set
            test_index = np.delete(test_index, np.argwhere(np.isin(test_index, self.is_augmented)))
            yield train_index, test_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to train the ML models."
    )

    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--delimiter", required=True)
    parser.add_argument("--ml-model", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    print(f"Starting to train the ML model(s): {args.ml_model}, target class: {args.target}")
    
    # Reading data
    data = pd.read_csv(args.train_csv, sep=args.delimiter, engine='python')
    target = args.target
    num_labels = len(data[target].unique())

    # Setting hyperparameters
    if args.ml_model == 'all':
        models = ['nb', 'svc', 'xgb']
    else:
        models = [args.ml_model]
    vectorizer = 'n_gram'
    parameters = {'nb': {'nb__alpha': [0.1, 0.3, 0.5, 0.7, 1.0],
                        'nb__fit_prior': [False, True]},
              'svc': {'svc__base_estimator__C': [0.1, 1, 10, 100],
                        'svc__base_estimator__gamma': ['auto', 'scale'],
                        'svc__base_estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                        'svc__base_estimator__class_weight': ['balanced', None]},
              'xgb': {'xgb__learning_rate': [0.01, 0.05, 0.1, 0.2],
                        'xgb__gamma': [0, 5, 10],
                        'xgb__subsample': [0.5, 0.7, 1],
                        'xgb__tree_method': ['exact', 'approx', 'hist'],
                        'xgb__objective': ['multi:softmax']},
              'n_gram': {'n_gram__ngram_range': [(3,3), (4,4)],
                          'n_gram__analyzer': ['word', 'char']}}
    functions = {'nb': ComplementNB(),
               'svc': CalibratedClassifierCV(SVC()),
               'xgb': xgboost.XGBClassifier(num_class=num_labels, seed=42),
               'n_gram': CountVectorizer()}


    cv_splitter = CustomSplit(data['fold'], data['is_augmented'])

    # Run the grid-search and train the final model (for each algorithm if you chose to train all the three ML algorithms)
    for model in models:
         start_time = time.time()
         print("---------------------------------------------------------")
         print(f"Training the model: {model}, target class: {args.target}")
         
         # Create the model directory
         model_name = str(model + '_' + args.target)
         model_dir = os.path.join(args.output_dir, model_name)
         os.makedirs(model_dir, exist_ok=True)
    
         # Set the pipeline and do grid-search
         params = {**parameters[vectorizer], **parameters[model]}
         vect_func = functions[vectorizer]
         model_func = functions[model]
         pipeline = Pipeline([(vectorizer, vect_func),
                             (model, model_func)])
         grid_search = GridSearchCV(pipeline, params, n_jobs=-1, verbose=1, cv=cv_splitter, refit=True)
         grid_search.fit(data['text_preprocessed_for_ml'], data[target])
         best_params = grid_search.best_params_
         print(f"Best accuracy: {grid_search.best_score_:.3f}")
         print("Best parameters set:")
         for param_name in sorted(params.keys()):
             print(f"   {param_name}: {best_params[param_name]}") #grid_search.best_estimator_.get_params()[param_name]}")
                   
         # For the final model, use multi:softprob so that the model is able to give probabilities for each class
         if args.ml_model == 'xgb':
            best_params['xgb__objective'] = "multi:softprob"
                   
         # Train the model with best parameters using all of the data
         print("Training the model with the best parameters using all training data.")
         pipeline.set_params(**best_params)
         pipeline.fit(data['text_preprocessed_for_ml'], data[target])
         
         model_filepath = os.path.join(model_dir, 'model.pkl')
         with open(model_filepath, 'wb') as f:
             pickle.dump(pipeline, f)
         print("Model saved as ", model_filepath)
             
         time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))
         print("Time elapsed for grid-search and training: ", time_elapsed)    
         print("---------------------------------------------------------")
         
    print('Training is ready.')
