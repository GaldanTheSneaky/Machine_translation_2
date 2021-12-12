from preprocess import Preprocessing
from model import NMTModel

INITIAL_LANGUAGE = 'ru'
TARGET_LANGUAGE = "en"
#RU_FILENAME = 'corpus.en_ru.1m.ru'
#EN_FILENAME = 'corpus.en_ru.1m.en'

RU_FILENAME = 'test_corpus.ru'
EN_FILENAME = 'test_corpus.en'


def runNMT():
    preprocessor = Preprocessing(initial_language=INITIAL_LANGUAGE,
                       target_language=TARGET_LANGUAGE,
                       initial_language_file=RU_FILENAME,
                       target_language_file=EN_FILENAME)

    initial_vocabulary, target_vocabulary, train_data, test_data = preprocessor.preprocess(
        chunk_size=50000, initial_stage=0, seq_length=15)
    test_data = ([train_data[0][2]], [train_data[2][2]])

    model = NMTModel(initial_vocabulary, target_vocabulary, use_attention=True)
    model.build()
    model.train(train_data, test_data, 30)

def runNSA():
    pass


if __name__ == "__main__":
    runNSA()