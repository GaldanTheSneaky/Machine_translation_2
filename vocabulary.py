class Vocabulary:
    PAD_token = 0  # Used for padding short sentences
    SOS_token = 1  # Start-of-sentence token
    EOS_token = 2  # End-of-sentence token

    def __init__(self, task): #tasks: MT, SA
        if task == "MT":
            self.word2index = {'_PAD_': self.PAD_token, 'START_': self.SOS_token, '_END': self.EOS_token}
            self.word2count = {'_PAD_': 0, 'START_': 0, '_END': 0}
            self.index2word = {self.PAD_token: '_PAD_', self.SOS_token: 'START_', self.EOS_token: '_END'}
            self.num_words = 3
        else:
            self.word2index = {}
            self.word2count = {}
            self.index2word = {}
            self.num_words = 0

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