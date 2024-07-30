#!/bin/bash

# Install Voikko and the morpho dictionary for Voikko
sudo apt-get install libvoikko-dev python-libvoikko voikko-fi
mkdir ./finnish/voikko-dictionary
wget -P ./finnish/voikko-dictionary/ https://www.puimula.org/htp/testing/voikko-snapshot-v5/dict-morpho.zip
unzip ./finnish/voikko-dictionary/dict-morpho.zip -d "./finnish/voikko-dictionary/"
rm ./finnish/voikko-dictionary/dict-morpho.zip

# Edit spacy_fi_experimental_web_md package source code to use the morpho dictionary
destinationDirectory=$(python ./finnish/find_fi.py)
cp -v ./finnish/modified_fi/fi.py $destinationDirectory"fi.py"
