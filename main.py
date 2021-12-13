from preprocess import Preprocessing
from MT_model import NMTModel
from SA_model import SAModel
import pandas as pd

INITIAL_LANGUAGE = 'ru'
TARGET_LANGUAGE = "en"

SA_FILENAME = "IMDB.csv"
SA_TEST_FILENAME = "test_SA_data.text"
RU_FILENAME = 'test_corpus.ru'
EN_FILENAME = 'test_corpus.en'


def load_tweet():
    df = pd.read_csv('Tweets.csv', sep=',')
    tweet_df = df[['text', 'airline_sentiment']]
    tweet_df = tweet_df[tweet_df['airline_sentiment'] != 'neutral']
    data = tweet_df
    return data[data.columns[0]].tolist(), data[data.columns[1]].tolist()


def run_main_SA():
    preprocessor = Preprocessing(initial_data="en",
                                 initial_data_file=SA_FILENAME,
                                 task="SA")

    initial_data, target_data = load_tweet()
    x_train, x_vocab, y_train, y_vocab = preprocessor.run(initial_stage=1, initial_corpus=initial_data, target_corpus=target_data)

    model = SAModel([x_train, y_train], y_vocab)
    model.build_model()
    model.train()


def run_main_MT():
    preprocessor = Preprocessing(initial_data=INITIAL_LANGUAGE,
                                 target_data=TARGET_LANGUAGE,
                                 initial_data_file=RU_FILENAME,
                                 target_data_file=EN_FILENAME,
                                 task="MT")

    initial_data, initial_vocab, target_data, target_vocab = preprocessor.run("txt", initial_stage=3)
    model = NMTModel(initial_vocab, target_vocab, use_attention=True)
    model.build()
    test_data = ([initial_data[2]], [target_data[1][2]])
    model.train([initial_data, target_data[0], target_data[1]], test_data, 30)


if __name__ == "__main__":
    run_main_SA()
