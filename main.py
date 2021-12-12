from preprocess import Preprocessing
from model import NMTModel
from SA_model import SAModel

INITIAL_LANGUAGE = 'ru'
TARGET_LANGUAGE = "en"
#RU_FILENAME = 'corpus.en_ru.1m.ru'
#EN_FILENAME = 'corpus.en_ru.1m.en'

SA_FILENAME = "IMDB.csv"
SA_TEST_FILENAME = "test_SA_data.text"
RU_FILENAME = 'test_corpus.ru'
EN_FILENAME = 'test_corpus.en'


def runNMT():
    preprocessor = Preprocessing(initial_language=INITIAL_LANGUAGE,
                       target_language=TARGET_LANGUAGE,
                       initial_language_file=RU_FILENAME,
                       target_language_file=EN_FILENAME)

    initial_vocabulary, target_vocabulary, train_data, test_data = preprocessor.preprocess_NMT(
        chunk_size=50000, initial_stage=0, seq_length=15)
    test_data = ([train_data[0][2]], [train_data[2][2]])

    model = NMTModel(initial_vocabulary, target_vocabulary, use_attention=True)
    model.build()
    model.train(train_data, test_data, 30)

def runNSA():
    preprocessor = Preprocessing(initial_language="en",
                       target_language=TARGET_LANGUAGE,
                       initial_language_file=SA_FILENAME,
                       target_language_file=EN_FILENAME)
    x_train, y_train, dict = preprocessor.preprocess_SA()

    model = SAModel([x_train, y_train], dict)
    model.build_model()
    model.train()

def run_main_SA():
    preprocessor = Preprocessing(initial_language="en",
                                 target_language=TARGET_LANGUAGE,
                                 initial_language_file=SA_FILENAME,
                                 target_language_file=EN_FILENAME,
                                 task="SA")

    x_train, x_vocab, y_train, y_vocab = preprocessor.run("csv")

    print(x_vocab.word2index)
    print(y_vocab.word2index)

    model = SAModel([x_train, y_train], y_vocab)
    model.build_model()
    model.train()


def run_main_MT():
    preprocessor = Preprocessing(initial_language=INITIAL_LANGUAGE,
                       target_language=TARGET_LANGUAGE,
                       initial_language_file=RU_FILENAME,
                       target_language_file=EN_FILENAME,
                       task="MT")

    initial_data, initial_vocab, target_data, target_vocab = preprocessor.run("txt")
    print(initial_data[0])
    print(target_data[0][0])
    print(target_data[1][0])
    model = NMTModel(initial_vocab, target_vocab, use_attention=True)
    model.build()
    test_data = ([initial_data[2]], [target_data[1][2]])
    model.train([initial_data, target_data[0], target_data[1]], test_data, 30)



if __name__ == "__main__":
    run_main_SA()