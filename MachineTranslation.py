import numpy as np
import keras
import nltk
import pickle
import string
from tqdm import tqdm


# INITIAL LANGUAGE = RU
# TARGET LANGUAGE = EN


def load_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def save_data(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)


class Vocabulary:
    PAD_token = 0  # Used for padding short sentences
    SOS_token = 1  # Start-of-sentence token
    EOS_token = 2  # End-of-sentence token

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.PAD_token: "_PAD_", self.SOS_token: "START_", self.EOS_token: "_END"}
        self.num_words = 3
        self.num_sentences = 0
        self.longest_sentence = 0

    def add_word(self, word):
        if word not in self.word2index:
            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # Word exists; increase word count
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        sentence_len = 0
        for word in sentence.split(' '):  # bit lame
            sentence_len += 1
            self.add_word(word)
        if sentence_len > self.longest_sentence:
            # This is the longest sentence
            self.longest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1

    def convert_sentence(self, sentence):
        converted_sent = []
        for word in sentence.split(' '):
            converted_sent.append(self.to_index(word))

        return converted_sent

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        return self.word2index[word]


class Translator:
    def __init__(self, initial_language: str, target_language: str, initial_language_file, target_language_file):
        self._initial_language = initial_language
        self._target_language = target_language
        self._initial_language_file = initial_language_file
        self._target_language_file = target_language_file
        self._initial_vocabulary = Vocabulary(f'{initial_language}')
        self._target_vocabulary = Vocabulary(f'{target_language}')

    def __load_corpus(self, filename, chunk_size, encoding='utf-8'):
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

    def clear_corporus(self, corpus, target_language=False):
        if target_language:
            corpus = ['START_ ' + sentence.lower().strip().translate(str.maketrans('', '', string.punctuation)) +
                      " _END" for sentence in tqdm(corpus)]
            save_data(f'cleaned_corpora.txt.{self._target_language}', corpus)
        else:
            corpus = [sentence.lower().strip().translate(str.maketrans('', '', string.punctuation))
                      for sentence in tqdm(corpus)]
            save_data(f'cleaned_corpora.txt.{self._initial_language}', corpus)

        return corpus

    def create_text_representation(self, corpus, seq_length, target_language=False):
        for i in tqdm(range(len(corpus))):
            if target_language:
                corpus[i] = self._target_vocabulary.convert_sentence(corpus[i][:])
            else:
                corpus[i] = self._initial_vocabulary.convert_sentence(corpus[i][:])

            sent_len = len(corpus[i])
            if sent_len >= seq_length:
                corpus[i] = corpus[i][:seq_length]
            else:
                for j in range(seq_length - sent_len):
                    corpus[i].append(0)

    def preprocess(self, chunk_size=0, initial_stage=1, seq_length=64):  # ADD ENUM WITH STAGES

        if initial_stage <= 1:
            print("STAGE ONE - CLEANING DATA")
            initial_corpus = self.__load_corpus(self._initial_language_file, chunk_size)
            target_corpus = self.__load_corpus(self._target_language_file, chunk_size)
            print("INITIAL CORPUS CLEANING...")
            cleaned_initial_corpus = self.clear_corporus(initial_corpus, target_language=False)
            print("TARGET CORPUS CLEANING...")
            cleaned_target_corpus = self.clear_corporus(target_corpus, target_language=True)
            print("STAGE ONE COMPLETE")

        if initial_stage <= 2:
            print("STAGE TWO - CREATING VOCABULARIES")
            if initial_stage > 1:
                print("LOADING DATA...")
                cleaned_initial_corpus = load_data(f'cleaned_corpora.txt.{self._initial_language}')
                cleaned_target_corpus = load_data(f'cleaned_corpora.txt.{self._target_language}')

            print("CREATING INITIAL LANGUAGE VOCABULARY...")
            for sentence in tqdm(cleaned_initial_corpus):
                self._initial_vocabulary.add_sentence(sentence)

            print("CREATING TARGET LANGUAGE VOCABULARY...")
            for sentence in tqdm(cleaned_target_corpus):
                self._target_vocabulary.add_sentence(sentence)

            print("SAVING VOCABULARY...")
            save_data(f'vocabulary.{self._initial_language}', self._initial_vocabulary)
            save_data(f'vocabulary.{self._target_language}', self._target_vocabulary)

            print("STAGE TWO COMPLETE")

        if initial_stage <= 3:
            print("STAGE THREE - CREATING TEXT REPRESENTATION")

            if initial_stage > 2:
                print("LOADING DATA...")
                cleaned_initial_corpus = load_data(f'cleaned_corpora.txt.{self._initial_language}')
                cleaned_target_corpus = load_data(f'cleaned_corpora.txt.{self._target_language}')
                self._initial_vocabulary = load_data(f'vocabulary.{self._initial_language}')
                self._target_vocabulary = load_data(f'vocabulary.{self._target_language}')

            print("CREATING INITIAL TEXT REPRESENTATION")
            self.create_text_representation(cleaned_initial_corpus, seq_length, target_language=False)
            initial_text_representation = np.array(cleaned_initial_corpus, dtype=object)
            save_data(f'text_representatiom.{self._initial_language}', initial_text_representation)

            print("CREATING TARGET TEXT REPRESENTATION")
            self.create_text_representation(cleaned_target_corpus, seq_length, target_language=True)
            target_text_representation = np.array(cleaned_target_corpus, dtype=object)
            save_data(f'text_representatiom.{self._target_language}', target_text_representation)



