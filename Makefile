.ONESHELL:

SHELL := /bin/bash

# Change the desired Conda environment name here
ENV_NAME := mri_protocoling

CONDA_BASE != conda info --base | tr -d '\n' | cat - <(echo "/etc/profile.d/conda.sh") | cat

PYTHON := python3
PIP := pip

FOLDS = 5
DELIMITER := "|" # delimiter used when reading .csv-files
ORIG_TRAIN_DATA := ./data/train.csv
EDITED_TRAIN_DATA:= ./data/edited_train.csv
OUTPUT_DIR := ./outputs

# If using in another language than Finnish, replace these paths
# The Spacy model (used in preprocessing the data). Models for other languages can be found here: https://spacy.io/usage/models
SPACY_PATH = "spacy_fi_experimental_web_md" 
# The pretrained BERT model. Other available models from HuggingFace can be found here: https://huggingface.co/models
BERT_PATH = "TurkuNLP/bert-base-finnish-cased-v1"

# Default values (can be overridden)
ML_MODEL ?= all
TARGET ?= protocol

# Possible values to give
ML_MODELS := nb svc xgb all
TARGETS := protocol contrast

TEST_CSV := data/test.csv

# The model to use for testing. Default value (bert) can be overriden.
TEST_MODEL ?= bert 
TEST_MODELS := bert nb svc xgb

# Check if the variables are valid
ifeq ($(filter $(ML_MODEL),$(ML_MODELS)),)
$(error Invalid ML_MODEL. Choose one from: $(ML_MODELS))
endif

ifeq ($(filter $(TARGET),$(TARGETS)),)
$(error Invalid TARGET. Choose one from: $(TARGETS))
endif

ifeq ($(filter $(TEST_MODEL),$(TEST_MODELS)),)
$(error Invalid TEST_MODEL. Choose one from: $(TEST_MODELS))
endif


# Create conda environment, activate it, and download Spacy model if not using the Finnish experimental model
create_conda_env:
	echo "name: " | tr -d '\n'| cat - <(echo ${ENV_NAME}) create_environment_template.yml > create_environment.yml
	conda env create -f create_environment.yml
	
	source ${CONDA_BASE}
	conda activate ${ENV_NAME}
	
	@if [ "$(SPACY_PATH)" != "spacy_fi_experimental_web_md"  ]
	then 
		${PYTHON} -m spacy download ${SPACY_PATH}
	fi
	
# Voikko installation and editing the Finnish experimental Spacy model to use a larger Voikko dictionary
install_finnish: 
	source ${CONDA_BASE}
	conda activate ${ENV_NAME}		
	
	chmod 755 finnish/install-voikko-and-edit-fi-experimental.sh
	./finnish/install-voikko-and-edit-fi-experimental.sh

# Preprocessing
split_to_folds:
	PYTHONHASHSEED=0 ${PYTHON} -m preprocessing.split_to_folds  \
			--folds=${FOLDS} \
			--input-csv=${ORIG_TRAIN_DATA} \
			--delimiter=${DELIMITER} \
			--target=${TARGET} \
			--output-csv=${EDITED_TRAIN_DATA} 

augmentation:
	PYTHONHASHSEED=0 ${PYTHON} -m preprocessing.augmentation \
			--train-csv=${EDITED_TRAIN_DATA} \
			--delimiter=${DELIMITER} \
			--target=${TARGET} \
			--spacy-path=${SPACY_PATH}

preprocess_data:
	PYTHONHASHSEED=0 ${PYTHON} -m preprocessing.preprocess_data \
			--train-csv=${EDITED_TRAIN_DATA} \
			--delimiter=${DELIMITER}  \
			--spacy-path=${SPACY_PATH}


# Training
train_ml:
	PYTHONHASHSEED=0 ${PYTHON} -m train_ml \
			--train-csv=${EDITED_TRAIN_DATA} \
			--delimiter=${DELIMITER} \
			--ml-model="${ML_MODEL}" \
			--target="${TARGET}" \
			--output-dir=${OUTPUT_DIR}

train_bert:
	PYTHONHASHSEED=0 ${PYTHON} -m train_bert \
			--train-csv=${EDITED_TRAIN_DATA} \
			--delimiter=${DELIMITER} \
			--bert-path=${BERT_PATH} \
			--folds=${FOLDS} \
			--target="${TARGET}" \
			--output-dir=${OUTPUT_DIR}

# Evaluation
evaluate:
	PYTHONHASHSEED=0 ${PYTHON} -m evaluate \
			--test-csv=${TEST_CSV} \
			--delimiter=${DELIMITER} \
			--spacy-path=${SPACY_PATH} \
			--bert-path=${BERT_PATH} \
			--test-model=${TEST_MODEL} \
			--output-dir=${OUTPUT_DIR}

# Clean
clean:
	rm -rf ${OUTPUT_DIR}
	@mkdir -p ${OUTPUT_DIR}

.PHONY: create_conda_env \
		install_finnish \
		split_to_folds \
		augmentation \
		preprocess_data \
		train_ml \
		train_bert \
		evaluate \
		clean
