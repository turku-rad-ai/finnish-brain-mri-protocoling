# Automated Protocoling of Finnish Emergency Brain MRI Referrals

This is the source code for our study "Artificial Intelligence for Automated Protocoling of Emergency Brain MRI Referrals" (not published yet). The code can be used to train and evaluate machine learning models and/or the BERT model to predict a suitable imaging protocol for medical imaging referrals and whether to use contrast agent or not. The code is designed primarily for Finnish language. It can be edited to work for other languages, except for the augmentation part which is only for Finnish language. The repository includes only the source code, not trained model weights.

## Requirements for the data

The training and testing data should be downloaded into the `data/` folder with names `train.csv` and `test.csv`. The delimiter for the csv-files is the vertical line | but the delimiter can be redefined in the Makefile. The data should contain columns named "text", "protocol", and "contrast". The "protocol" label should be encoded into numerical format (numbers from 0 to N-1 if there are N different classes). The "contrast" label should be binary and in numerical format (0 or 1). 

## Language

By default, the code is designed for Finnish language. If your data is in other language, please change these variables first in the Makefile: 

- `SPACY_PATH`, the spaCy model used in preprocessing the text data. By default, this is set as "spacy_fi_experimental_web_md" (Finnish experimental model). Available spaCy models can be found here: <https://spacy.io/usage/models>
- `BERT_PATH`, the pretrained BERT model and tokenizer. The default value for this is "TurkuNLP/bert-base-finnish-cased-v1". Available BERT models can be found here: <https://huggingface.co/models>

Please note that the augmentation is specifically for Finnish and will not work in any other language. However, using augmentation is not required for training the models, and the augmentation step can be skipped.

## Environment installation

The code was tested on Ubuntu (22.04.4 LTS) using Anaconda (v. 4.12.0). The BERT model is trained with Pytorch (v. 1.10) using GPU support (CUDA 10.2). The software package requirements and version info can be found in the `create_environment_template.yml`.

To create the conda environment, run the following command. By default, the environment will be named "mri_protocoling". If you wish to use another name, you can change the variable `ENV_NAME` in the Makefile before creating the environment.

```
make create_conda_env
```

Then, if using data in Finnish, run the following command:

```
make install_finnish
```

This installs Voikko (a linguistic analysis software for Finnish, https://voikko.puimula.org/) and the "morpho" dictionary, which has a larger vocabulary (including medical terms) than the default "standard" dictionary. Notice that the installation requires root privileges. We also utilized the Finnish experimental spaCy model created by Antti Ajanki (https://github.com/aajanki/spacy-fi, MIT License). We edited the package's source code in order for it to utilize the "morpho" dictionary. make install_finnish overwrites the original `fi.py` file in the spacy_fi_experimental_web_md package's folder with our edited version of it. No other changes were made in the package's source code. 

## Preprocessing the data

Training is done using 5-fold cross-validation. To split the training data into folds, stratified by the protocol class, run the following command. The output is saved as a new file called `edited_train.csv`. If you want to do the split stratified by the contrast class, pass the argument `TARGET=contrast` to the command.

```
make split_to_folds
```

Then, if using Finnish data, you can do data augmentation. This is not a mandatory step. The augmentation process can take some time. By default, this creates 5 augmented versions of each original instance in the training set, but the code can be customized to augment some protocol classes more than the others (see the comments in `preprocessing/augmentation.py`). See also "Notes about the augmentation".

```
make augmentation
```

Finally, regardless of whether you used augmentation or not, run this to clean the referrals (simplifying whitespaces) and to do preprocessing required for machine learning models (lemmatization and changing into lowercase).

```
make preprocess_data
```

## Training the models

The models are trained separately to predict the protocol and to predict the need for contrast agent. 

For predicting the correct protocol, the machine learning (ML) models can be trained using the command:

```
make train_ml
```

By default, this runs the grid-search and training for all three different ML models (Naive Bayes, Support Vector Classifier, and XGBoost). If you want to train only one of the algorithms, you can pass the argument `ML_MODEL` for the command. Possible options for the `ML_MODEL` are `nb` for Naive Bayes, `svc` for Support Vector Classifier, or `xgb` for XGBoost.

```
make train_ml ML_MODEL=nb
```

If you want to train the BERT model for protocol prediction, run the following command. 

```
make train_bert
```

If you want to train the models for predicting the need for contrast agent, add the argument `TARGET=contrast` to the training command:

```
make train_ml TARGET=contrast
make train_bert TARGET=contrast
```

The trained models are placed in the folder `outputs/`.

## Evaluating the models

To evaluate the models on the test set (which should be placed in the `data/` folder and named as `test.csv`), run the following command. The variable `TEST_MODEL` in Makefile determines the default model algorithm used for evaluation. The model can also be selected by passing its name as an argument. The evaluation is run for both the protocol model and the contrast model at the same time, so both versions need to be trained before evaluation. The possible values for `TEST_MODEL` are `bert`, `nb`, `svc`, or `xgb`. For example, to run the evaluation with the BERT models, run the following command:

```
make evaluate TEST_MODEL=bert
```

This prints the classification reports for the models and saves the predictions for each referral in csv format in the `outputs/` folder.

## Notes about augmentation

The augmentation process uses a custom dictionary `word_replacements.csv` for replacing certain (mostly medical) words with their synonyms and abbreviations or other words that belong loosely in the same category (e.g. different types of gliomas). The dictionary also contains medical slang terms often used in Finnish imaging referrals or different spellings for some terms. The dictionary is designed for brain MRI referrals, so the majority of the words are related to neurological issues and thus it may not be suited for augmenting other kind of referrals. Please also note, that the dictionary is not exhaustive and the augmented referrals might end up containing medical inaccuracies or contradictions. The purpose of using a custom dictionary was to include more variation in the augmented referrals used for training, not to create medically accurate referrals (as this would have been a much more complex task, both linguistically and programmatically). 

Other augmentation methods used in the code are: backtranslation, swapping the order of sentences or removing random sentences, swapping or removing random characters, and randomly increasing or decreasing numbers.
