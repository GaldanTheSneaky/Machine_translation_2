from MachineTranslation import Translator

INITIAL_LANGUAGE = 'ru'
TARGET_LANGUAGE = "en"
RU_FILENAME = 'corpus.en_ru.1m.ru'
EN_FILENAME = 'corpus.en_ru.1m.en'


def main():
    model = Translator(initial_language=INITIAL_LANGUAGE,
                       target_language=TARGET_LANGUAGE,
                       initial_language_file=RU_FILENAME,
                       target_language_file=EN_FILENAME)
    model.preprocess(chunk_size=0, initial_stage=0)



if __name__ == "__main__":
    main()