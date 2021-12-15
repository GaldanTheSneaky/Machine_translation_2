import numpy as np
from keras import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam


class SAModel:
    def __init__(self, training_data, source_dict):
        self.x_train = training_data[0]
        self.y_train = training_data[1]
        self.hidden_size = 200
        self.embedding_size = 100
        self.hidden_dropout_rate = 0.2
        self.embedding_dropout_rate = 0.2
        self.batch_size = 100
        self.max_target_step = 30
        self.vocab_size = len(source_dict.word2index)
        self.source_dict = source_dict
        self.train_model = None
        self.encoder_model = None
        self.decoder_model = None

    def build(self):

        model = Sequential()
        model.add(Embedding(self.vocab_size, 200, input_length=len(self.x_train[0]), mask_zero=True))
        model.add(Dropout(self.embedding_dropout_rate))

        model.add(Bidirectional(LSTM(256, return_sequences=True)))
        model.add(Dropout(self.hidden_dropout_rate))
        model.add(BatchNormalization())

        model.add(Bidirectional(LSTM(128)))
        model.add(Dropout(self.hidden_dropout_rate))
        model.add(BatchNormalization())

        model.add(Dense(64, activation='relu'))
        model.add(Dropout(self.hidden_dropout_rate))
        model.add(BatchNormalization())

        model.add(Dense(self.vocab_size, activation='softmax'))

        opt = Adam(learning_rate=0.0001, decay=1e-6)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        print(model.summary())
        self._model = model

    def train(self, batch_size=300, epochs=30, verbose=1):
        x_train = self.x_train
        x_train = np.asarray(x_train).astype('int32')
        y_train = self.y_train
        y_train = np.asarray(y_train).astype('int32')

        y_train = np.reshape(y_train, (np.shape(y_train)[0], 1))

        self._model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.2,
                        verbose=verbose)

        self._model.save('model')
        print("Finish")



