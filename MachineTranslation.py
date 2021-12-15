from preprocess import Preprocessing
from MT_model import NMTModel


class MachineTranslation:
    def __init__(self, initial_data_file=None,
                 target_data_file=None,
                 initial_data=None,
                 target_data=None,
                 prepr_initial_stage=0):
        self.preprocessor = Preprocessing(initial_data_file=initial_data_file, target_data_file=target_data_file,
                                          task="MT")
        self.prepr_initial_stage = prepr_initial_stage
        self.initial_data=initial_data
        self.target_data=target_data

    def run(self):
        initial_data, initial_vocab, target_data, target_vocab = self.preprocessor.run(
            data_extension="txt",
            initial_stage=self.prepr_initial_stage,
            initial_corpus=self.initial_data,
            target_corpus=self.target_data)

        self.model = NMTModel(initial_vocab, target_vocab, use_attention=True)
        self.model.build()
        test_data = ([initial_data[2]], [target_data[1][2]])
        self.model.train([initial_data, target_data[0], target_data[1]], test_data, 30)
