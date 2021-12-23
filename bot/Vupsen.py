from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten, Reshape, Conv1D, GlobalMaxPooling1D, Bidirectional
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow import constant
import pandas as pd
import numpy as np
from Preparator import Preparator
from Pupsen import Pupsen

from sklearn.model_selection import train_test_split

class Vupsen(Preparator):

    def __init__(self, preparator) -> None:
        self.ft_model = preparator.ft_model
        self.maxlen = 8 
        self.filters = 200
        self.kernel_size = 3
        self.embeding_dims = 300
        self.batch_size = 13
        self.epoch = 20
        self.build_model()

    def build_model(self):
        self.model = Sequential()   

        self.model.add(Bidirectional(LSTM(8, return_sequences=True, input_shape = (self.maxlen, 300))))

        self.model.add(Flatten())

        self.model.add(Dropout(0.2))

        self.model.add(Dense(256, activation="tanh"))

        self.model.add(Dense(128, activation="sigmoid"))

        self.model.add(Dense(64, activation="relu"))

        self.model.add(Dropout(0.2))

        self.model.add(Dense(self.maxlen, activation="softmax"))

        self.model.compile(loss = "categorical_crossentropy",
                            optimizer = "adam") #"rmsprop", KLDivergence, categorical_crossentropy    

    def learn(self):
        data = pd.read_csv("текст.csv", delimiter=",")
        data = data[data.важность != 0]
        text_rare = data["Текст"]
        answer = data["ответ"].to_numpy()
        answer_ = data["ответ"].to_numpy()
        answer_ = np.array([self.ft_model.wv[word] for word in answer_])
        text_rare = [self.clear_text(text) for text in text_rare]
        answer = [self.clear_text(text) for text in answer]
        text_tokenize = [self.text_tokenize(text) for text in text_rare]
        answer = [self.text_tokenize(text) for text in answer]
        for text in range(len(answer)):
            answer[text] = self.classify_text(text_tokenize[text], answer[text])
        
        index = np.where(np.array(answer) >= self.maxlen)
        answer = np.delete(answer, index)
        text_tokenize = np.delete(text_tokenize, index)

        text_ready = [self.text_vectorize(text, lengh = 25) for text in text_tokenize]
        text_np = np.array([text for text in text_ready])
        text_f = np.array([])
        data_len = len(answer)

        print(np.max(answer))
        print(np.mean(answer))
        print(data_len)

        answer = to_categorical(answer, num_classes=self.maxlen)
        answer = constant(answer, shape=[data_len, self.maxlen])
        for i in range(len(text_np)):
            text_f = np.append(text_f, text_np[i])
        text_f = text_f.reshape(data_len,25,300)

        X = text_f[:380]
        Y = answer[:380]
        x = text_f[380:]
        y = answer[380:]

        self.model.fit(text_f, answer, 
                        batch_size = self.batch_size,
                        epochs = self.epoch,
                        # validation_data = (x,y)
        )

    def predict(self, input_text):
        text = self.clear_text(input_text)
        text_token = self.text_tokenize(text)
        text = self.text_vectorize(text_token, lengh = 25)
        answer = self.model.predict(text)
        return text_token[np.argmax(answer[0][:len(text_token)])], np.amax(answer)