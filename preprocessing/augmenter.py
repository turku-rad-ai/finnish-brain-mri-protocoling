# -*- coding: utf-8 -*-

import os
import re
import csv
import random as rnd
import spacy
from libvoikko import Voikko
from voikko import inflect_word
import nlpaug.augmenter.char as nlpaug_char
import nlpaug.augmenter.word as nlpaug_word
import nlpaug.flow as nlpaug_flow

def create_word_replacement_dict():
  """Create a dictionary of synonyms and other word replacements from a list of lists."""
  with open('./preprocessing/word_replacements.csv', 'r') as read_obj:
    csv_reader = csv.reader(read_obj)
    word_lists = list(csv_reader)

  word_replacements = {}

  for l in word_lists:
    # add each word to the word replacement dictionary and set all it's other word replacement options as the value
    for word in l:
      word_list = l.copy()
      word_list.remove(word)
      word_replacements.setdefault(word.lower(), []).extend(x for x in word_list if x not in word_replacements[word.lower()])

      # this is to ensure the augmenting function works with word replacements that are combination of multiple words
      if ' ' in word:
        first_word = word.split()[0].lower()
        word_replacements.setdefault(first_word, []).append(first_word)

  return word_replacements

class CustomTokenizer:
  """Custom tokenizer"""
  def __init__(self, nlp):
    self.orig_whitespaces = []
    self.nlp = nlp
   
  def tokenizer(self, text):
    """Tokenize text"""
    doc = self.nlp(text)
    tokens = [token.text for token in doc if token.text.strip()]
    self.orig_whitespaces.extend([token.whitespace_ for token in doc])
    return tokens

  def reverse_tokenizer(self, tokens):
    """Join tokens together with original whitespaces"""
    return ''.join([tokens[i] + self.orig_whitespaces[i] for i in range(len(tokens))]).strip()

class Augmenter():
  """Augmenter class"""
  def __init__(self, spacy_path):
    self.dictionary = create_word_replacement_dict()
    self.nlp = spacy.load(spacy_path)
    self.v = Voikko('fi-x-morpho', path="./finnish/voikko-dictionary/5/")
    self.backtranslator = nlpaug_word.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-tc-big-fi-en', to_model_name='Helsinki-NLP/opus-mt-tc-big-en-fi', max_length=1200)

  def augment(self, text):
    """Augment text at token, sentence and character level"""
    modified = ""
    modified = self.__token_augmenter(text)
    modified = self.__sent_augmenter(modified)
    modified = self.__char_augmenter(modified)

    return modified

  def __random_execute(self, chance):
    """Execute an action with predetermined chance"""
    return rnd.random() < chance

  def __get_new_word(self, word):
    """Get random word replacement for the word"""
    new_word = rnd.choice(self.dictionary.get(word))
    return new_word

  def __edit_word(self, new_word, sentence_start, whitespace):
    """Add whitespace and capitalize if it is the start of the sentence"""
    if sentence_start and new_word[0].islower():
      new_word = new_word.capitalize()
    new_word = new_word + whitespace
    return new_word

  def __get_edited_new_word(self, word, sentence_start, whitespace):
    """Get and edit new word"""
    new_word = self.__get_new_word(word)
    new_word = self.__edit_word(new_word, sentence_start, whitespace)
    return new_word

  def __get_inflected_word(self, new_word, baseform, orig_word):
    """Inflect the new word in similar way than the original word"""
    # create word inflections for both words
    orig_inflection = inflect_word.generate_forms(baseform)
    new_inflection = inflect_word.generate_forms(new_word)

    # search which inflection is used in the original word and get that form for the new word
    if len(orig_inflection[0]) > 0 and len(new_inflection[0]) > 0:
      for infl in orig_inflection[0]:
        if orig_word == orig_inflection[0][infl][0]:
          new_word = new_inflection[0][infl][0]

    return new_word

  def __num_replacer(self, matchobj):
    """Replace numbers with a number within +/-10 from the original"""
    num = int(matchobj[0])
    lower_range = max(1, num-10)
    upper_range = num+10
    random_number = str(rnd.randint(lower_range, upper_range))

    return random_number

  def __swap_sents(self, sents, index1, index2):
    """Function to swap sentence order"""
    sents[index1], sents[index2] = sents[index2], sents[index1]
    return sents

  def __token_augmenter(self, text):
    """Token-level augmenter"""
    doc = self.nlp(text)
    skip_next_token = False
    modified = ""

    for ix, token in enumerate(doc):

      # if the word is already handled in the previous step (as a combination of two words), skip to the next word
      if skip_next_token:
        skip_next_token = False
        continue

      word = token.text.lower()

      # augment numbers
      word = re.sub('[0-9]+', self.__num_replacer, word)

      # augment words (custom word replacer)
      if word in self.dictionary:
        # if not the last word, then check if the word + next word combination can be found in the dictionary, and get a synonym/word replacement for the combination
        if ix != (len(doc)-1):
          combined = (word + ' ' + doc[ix+1].text.lower())
          if combined in self.dictionary:
            new_word = self.__get_edited_new_word(combined, token.is_sent_start, doc[ix+1].whitespace_)
            skip_next_token = True
          # if the combination is not found, then get the replacement for the first word
          else:
            new_word = self.__get_edited_new_word(word, token.is_sent_start, token.whitespace_)
        # if the word is the last word, no need to check for possible word combinations
        else:
          new_word = self.__get_edited_new_word(word, token.is_sent_start, token.whitespace_)

        modified += new_word

      # if the inflected form of the word is not in the word replacement list, try changing the word back to baseform and search again
      # for now, this is only done for single words and not for word combinations
      elif len(self.v.analyze(word)) > 0:
        baseform = self.v.analyze(word)[0][u"BASEFORM"]

        if baseform in self.dictionary:
          new_word = self.__get_new_word(baseform)
          new_word = self.__get_inflected_word(new_word, baseform, word)
          new_word = self.__edit_word(new_word, token.is_sent_start, token.whitespace_)
          modified += new_word

        else:
          modified += self.__edit_word(word, token.is_sent_start, token.whitespace_)

      # no word replacements for the word, save the original word
      else:
        modified += token.text_with_ws

    return modified

  def __sent_augmenter(self,text):
    """Sentence-level augmenter"""
    doc = self.nlp(text)
    modified = ""

    sentences = list(doc.sents)
    sentences_number = len(sentences)

    # swap sentences (only if there is 4 or more sentences in the referral)
    if sentences_number >= 4:
      sentence_to_swap = rnd.choice(range(0, sentences_number))
      other_sentence = sentence_to_swap+1
      if sentence_to_swap == sentences_number-1:
        other_sentence = sentence_to_swap-1
      sentences = self.__swap_sents(sentences, sentence_to_swap, other_sentence)

    # do randomly: drop out the sentence, do back-translation, or keep it as it is
    for sentence in sentences:
      if sentences_number > 1 and self.__random_execute(0.05):
        continue
      elif self.__random_execute(0.5):
        modified += self.backtranslator.augment(sentence.text)
        modified += ' '
      else:
        modified += sentence.text_with_ws

    return modified

  def __char_augmenter(self, text):
    """Character-level augmenter"""
    modified = ""

    # tokenizer functions
    CustomTok = CustomTokenizer(self.nlp)
    tok = CustomTok.tokenizer
    rev_tok = CustomTok.reverse_tokenizer

    # augmenting functions (keyboard mistake, random character swap or delete)
    keyboard_error_aug = nlpaug_char.KeyboardAug(include_numeric=False, include_special_char=False, include_upper_case=False, aug_char_max=1, aug_word_p=0.05, tokenizer=tok, reverse_tokenizer=rev_tok)
    random_swap_aug = nlpaug_char.RandomCharAug(aug_char_max=1, aug_word_p=0.05, aug_word_max=1, action="swap", swap_mode='adjacent', tokenizer=tok, reverse_tokenizer=rev_tok)
    random_delete_aug = nlpaug_char.RandomCharAug(aug_char_max=1, aug_word_p=0.05, action="delete", tokenizer=tok, reverse_tokenizer=rev_tok)

    # use nlp-aug flow for doing the character augmentations
    flow = nlpaug_flow.Sometimes([keyboard_error_aug, random_swap_aug, random_delete_aug], aug_p=0.1)
    modified = flow.augment(text)

    return modified
