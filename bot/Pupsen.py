from tensorflow.keras.layers import Conv1D, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow import constant
import pandas as pd
import numpy as np
from Preparator import Preparator

# В тексте нет ничего важного - 0
# В тексте есть что-то важное - 1

class Pupsen(Preparator):

    def __init__(self, preparator) -> None:
        self.ft_model = preparator.ft_model
        self.maxlen = preparator.maxlen
        self.filters = 200
        self.kernel_size = 3
        self.embeding_dims = 300
        self.batch_size = 20
        self.epoch = 5
        self.build_model()

    def build_model(self):
        self.model = Sequential()

        self.model.add(Conv1D(
            self.filters,
            self.kernel_size,
            padding = "valid",
            activation = "relu",
            strides = 1,
            input_shape = (self.maxlen, 300),
        ))

        self.model.add(GlobalMaxPooling1D())

        self.model.add(Dense(250, activation="relu"))

        self.model.add(Dropout(0.2))

        self.model.add(Dense(1, activation="sigmoid"))

        self.model.compile(loss = "binary_crossentropy",
                            optimizer = "adam")

    def learn(self):
        data = pd.read_csv("текст.csv", delimiter= ",")
        text_rare = data["Текст"]
        warn = data["важность"].to_numpy()
        text_rare = [self.clear_text(text) for text in text_rare]
        text_tokenize = [self.text_tokenize(text) for text in text_rare]
        text_ready = [self.text_vectorize(text, lengh = self.maxlen) for text in text_tokenize]
        text_np = np.array([text for text in text_ready])
        text_f = np.array([])
        data_len = len(warn)
        for i in range(len(text_np)):
            text_f = np.append(text_f, np.array(text_np[i][0]))
        text_f = text_f.reshape(data_len,self.maxlen,300)
        self.model.fit(text_f, warn, 
                        batch_size = self.batch_size,
                        epochs = self.epoch)

    def predict(self, input_text):
        text = self.clear_text(input_text)
        text_token = self.text_tokenize(text)
        text = self.text_vectorize(text_token, lengh = self.maxlen)
        answer = self.model.predict(text)
        return answer