import numpy as np
import nltk
import pickle
import string
from tqdm import tqdm
import pandas as pd
from nltk.corpus import stopwords
import os
import random

from vocabulary import Vocabulary


def list_splitter(list_to_split, ratio):
    first_half = int(len(list_to_split) * ratio)
    return list_to_split[:first_half], list_to_split[first_half:]


def load_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def save_data(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)


class Preprocessing:
    def __init__(self, task, initial_data: str = None, target_data: str = None, initial_data_file=None, target_data_file=None):
        self._initial_language = initial_data
        self._target_language = target_data
        self._initial_data_file = initial_data_file
        self._target_data_file = target_data_file
        self._initial_vocabulary = Vocabulary(f'{initial_data}', task)
        self._target_vocabulary = Vocabulary(f'{target_data}', task)
        self._task = task

    def _load_by_line(self, filename, chunk_size, encoding='utf-8'):
        corpus = []
        with open(filename, encoding=encoding) as file:
            if chunk_size == 0:
                for line in file:
                    corpus.append(line)
            else:
                for line in file:
                    if chunk_size == 0:
                        break
                    else:
                        corpus.append(line)
                        chunk_size -= 1

        return corpus

    def _load_csv(self, filename):
        data = pd.read_csv(filename)
        return data[data.columns[0]].tolist(), data[data.columns[1]].tolist()

    def _load_tweet(self, filename):  # delete in the future
        df = pd.read_csv('Tweets.csv', sep=',')
        tweet_df = df[['text', 'airline_sentiment']]
        tweet_df = tweet_df[tweet_df['airline_sentiment'] != 'neutral']
        data = tweet_df
        return data[data.columns[0]].tolist(), data[data.columns[1]].tolist()

    def clear_corpus(self, corpus, save_file_type, target_language=False):
        if target_language:
            corpus = ['START_ ' + sentence.lower().strip().translate(str.maketrans('', '', string.punctuation)) +
                      " _END" for sentence in tqdm(corpus)]
            ###
            # corpus = [' '.join([word for word in sentence.split(' ') if word not in
            #           stopwords.words('english')]) for sentence in tqdm(corpus)]
            ###

            save_data(f'cleaned_corpus.txt.{save_file_type}', corpus)
        else:
            corpus = [sentence.lower().strip().translate(str.maketrans('', '', string.punctuation))
                      for sentence in tqdm(corpus)]

            # corpus = [' '.join([word for word in sentence.split(' ') if word not in
            #            stopwords.words('english')]) for sentence in tqdm(corpus)]

            save_data(f'cleaned_corpus.txt.{save_file_type}', corpus)

        return corpus

    def create_text_representation(self, corpus, seq_length, vocab, target_language=False):
        for i in tqdm(range(len(corpus))):
            if target_language:
                corpus[i] = vocab.convert_sentence(corpus[i][:])
            else:
                corpus[i] = vocab.convert_sentence(corpus[i][:])

            sent_len = len(corpus[i])
            if sent_len >= seq_length + 1:
                corpus[i] = corpus[i][:seq_length]
            else:
                for j in range(seq_length - sent_len):
                    corpus[i].append(0)

        return corpus

    def create_output_data(self, target_corpus):
        target_output_representation = []
        for i in range(len(target_corpus)):
            target_output_representation.append(target_corpus[i][1:])
            target_corpus[i] = target_corpus[i][:-1]

        target_input_representation = np.array(target_corpus, dtype=object)
        target_output_representation = np.array(target_output_representation, dtype=object)

        training_data = list(zip(target_input_representation, target_output_representation))
        random.shuffle(training_data)
        training_data = np.array(training_data)
        target_input_representation = training_data[:, 0]
        target_output_representation = training_data[:, 1]

        a, b = np.shape(target_output_representation)
        target_output_representation = np.reshape(target_output_representation, (a, b, 1))  # LAME

        return target_input_representation, target_output_representation

    def run(self, data_extension=None, chunk_size=0, initial_stage=0, seq_length=50, initial_corpus=None,
            target_corpus=None):  # data_extensions: txt(line by line), csv
        if initial_stage <= 0:
            print("STAGE ZERO - LOADING DATA")
            if data_extension == "txt":
                initial_corpus = self._load_by_line(self._initial_data_file, chunk_size)
                target_corpus = self._load_by_line(self._target_data_file, chunk_size)
            elif data_extension == "csv":
                initial_corpus, target_corpus = self._load_csv(self._initial_data_file)
            else:
                raise (Exception("Invalid file extension"))

        initial_data, initial_vocab = self.preprocess(initial_corpus, initial_stage=initial_stage,
                                                      seq_length=seq_length, if_target=False)
        target_data, target_vocab = self.preprocess(target_corpus, initial_stage=initial_stage, seq_length=seq_length,
                                                    if_target=True)

        return initial_data, initial_vocab, target_data, target_vocab

    def preprocess(self, corpus, initial_stage, seq_length, if_target: bool):  # tasks: SA, MT
        if if_target:
            save_file_type = "target"
        else:
            save_file_type = "initial"

        if initial_stage <= 1:
            print("STAGE ONE - CLEANING DATA")
            print("INITIAL CORPUS CLEANING...")
            if self._task == "SA" and if_target:
                cleaned_corpus = corpus
            else:
                cleaned_corpus = self.clear_corpus(corpus, target_language=if_target, save_file_type=save_file_type)

            print("STAGE ONE COMPLETE")

        if initial_stage <= 2:
            print("STAGE TWO - CREATING VOCABULARIES")
            if initial_stage > 1:
                print("LOADING DATA...")
                cleaned_corpus = load_data(f'cleaned_corpus.txt.{save_file_type}')

            print("CREATING INITIAL LANGUAGE VOCABULARY...")
            vocabulary = Vocabulary(f'{save_file_type}', self._task)
            for sentence in tqdm(cleaned_corpus):
                vocabulary.add_sentence(sentence)

            print("SAVING VOCABULARY...")
            save_data(f'vocabulary.{save_file_type}', vocabulary)

            print("STAGE TWO COMPLETE")

        if initial_stage <= 3:
            print("STAGE THREE - CREATING TEXT REPRESENTATION")

            if initial_stage > 2:
                print("LOADING DATA...")
                cleaned_corpus = load_data(f'cleaned_corpus.txt.{save_file_type}')
                vocabulary = load_data(f'vocabulary.{save_file_type}')

            print("CREATING TEXT REPRESENTATION...")
            if self._task == "SA" and if_target:
                seq_length = 1

            cleaned_corpus = self.create_text_representation(
                cleaned_corpus, seq_length, vocabulary, target_language=if_target)
            text_representation = np.array(cleaned_corpus, dtype=object)
            save_data(f'text_representation.{save_file_type}', text_representation)

            if self._task == "MT" and if_target:
                target_input_representation, target_output_representation = self.create_output_data(
                    cleaned_corpus)
                save_data(f'target_input_representation.{save_file_type}', target_input_representation)
                save_data(f'target_output_representation.{save_file_type}', target_output_representation)
                return [target_input_representation, target_output_representation], vocabulary
            else:
                return text_representation, vocabulary
