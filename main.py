import pandas as pd

from SentimentAnalysis import SentimentAnalysis
from MachineTranslation import MachineTranslation

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


def SA():
    initial_data, target_data = load_tweet()
    SA = SentimentAnalysis(initial_data=initial_data, target_data=target_data, prepr_initial_stage=1)
    SA.run()


def MT():
    MT = MachineTranslation(initial_data_file=RU_FILENAME, target_data_file=EN_FILENAME)
    MT.run()


if __name__ == "__main__":
    MT()
