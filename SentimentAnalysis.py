from preprocess import Preprocessing
from SA_model import SAModel


class SentimentAnalysis:
    def __init__(self, initial_data_file=None,
                 target_data_file=None,
                 initial_data=None,
                 target_data=None,
                 prepr_initial_stage=0):
        self.preprocessor = Preprocessing(initial_data_file=initial_data_file,
                                          target_data_file=target_data_file,
                                          task="SA")
        self.initial_data = initial_data
        self.target_data = target_data
        self.prepr_initial_stage = prepr_initial_stage

    def run(self):
        x_train, x_vocab, y_train, y_vocab = self.preprocessor.run(initial_stage=self.prepr_initial_stage,
                                                                   initial_corpus=self.initial_data,
                                                                   target_corpus=self.target_data)

        self.model = SAModel([x_train, y_train], y_vocab)
        self.model.build()
        self.model.train()
