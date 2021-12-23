import tweepy
import numpy as np
from tweepy import api
import string
from gensim.models.fasttext import FastText
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re


class Preparator:
    def __init__(self) -> None:
        self.answer = ""
        self.emotions = 0
        self.plot = ""
        self.proba = 0
        self.importance = 0
        self.maxlen = 280
        self.alphabet = "/абвгдеёжзийклмнопрстуфхцчшщъыьэюя.,- ?!"
        self.strat_token = 41.
        self.end_token = 42.
        self.fill_token = 40.

    def init_ft(self):
        self.ft_model = FastText.load_fasttext_format(model_file="cc.ru.300.bin")

    def get_random(self, theme: str):
        "Get random tweet by you twitter development acc"
        consumer_key = ""
        consumer_secret = ""
        acess_token = ""
        acess_token_secret = ""

        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(acess_token, acess_token_secret)
        api = tweepy.API(auth)

        cursor = tweepy.Cursor(
            api.search_tweets, q=theme, lang="ru", tweet_mode="extended"
        ).items(45)
        tweepy.Cursor(
            api.search_full_archive,
        )
        text = []
        for i in cursor:
            text += [i.full_text]
        return np.random.choice(text)

    def expand_text(self, text,lengh, ending=0.0):
        return text[:lengh] + [[ending] * 300] * (lengh - len(text))

    def text_tokenize(self,text):
        tokens = text.lower()
        tokens = word_tokenize(text)
        tokens = [i for i in tokens if (i not in string.punctuation)]
        stop_words = stopwords.words('russian')
        tokens = [i for i in tokens if (i not in stop_words)]
        return tokens

    def text_vectorize(self, text,lengh):
        vectors = [self.ft_model.wv[word] for word in text]
        vectors = [vectors[:lengh] + [[0.]*300]*(lengh - len(vectors))]
        return np.array(vectors)

    def clear_text(self, text):
        text = re.sub(r"\s+", " ", text.strip())
        text = re.sub(r"^(RT )?@.+?: ", "", text)
        text = re.sub(r"@.+? ", "", text)
        text = re.sub(r" [A-Za-z]+://.+ ", "", text)
        text = re.sub(r" [,-;»-)—(*+:«"]+://.+ ", "", text)
        text = text.lower()
        return text

    def alphabet_text(self, text, c = 0, lengh = 20):
        "Get vectorize text by letters"
        text = self.clear_text(text)
        lengh_normative = False
        text = text.lower()
        text_1 = []
        text_2 = [self.strat_token]
        for word in range(len(text)):
            for letter in range(len(text[word])):
                if  text[word][letter] in self.alphabet:
                    text_1 += [float(self.alphabet.index(text[word][letter]))]
                    text_2 += [float(self.alphabet.index(text[word][letter]))]
        text_2 += [self.end_token]
        text_1 += [self.end_token]
        if len(text_1) >= lengh:
            lengh_normative = True
        text_2 = text_2[: lengh] + [self.fill_token] * (lengh - len(text_2))
        text_1 = text_1[: lengh] + [self.fill_token] * (lengh - len(text_1))
        if lengh_normative:
            text_2[-1] = self.end_token
            text_1[-1] = self.end_token
        if c == 0:
            return np.array(text_2)
        else:
            return np.array(text_1)

    def to_alphabet(self, input):
        "from letters vector to string"
        answer = ""
        i = 0
        while i < len(input) and input[i] < len(self.alphabet):
            answer += self.alphabet[input[i]]
            i += 1
        return answer

    def classify_text(self, text, keyword):
        return text.index(keyword[0])

    def get_book(self,library):
        "get only dialoges from saved books"
        respond = []
        answer = []
        for book in library:
            print(book)
            with open(f'books/{book}.txt',  encoding = "UTF-8") as f:
                lines = f.readlines()
            dialoges = []
            for line in lines:
                if line[0] == "—" or line[0] == "–":
                    dialoges += [line]
                else:
                    for word in range(len(line)):
                        if line[word] == "\xa0" and (line[word - 1] == "—" or line[word - 1] == "–"):
                            dialoges += [line]
            for i in range(len(dialoges)):
                dialoges[i] = re.sub("\xa0", "",dialoges[i])
                dialoges[i] = re.sub("\n", "",dialoges[i])
                dialoges[i] = re.sub(r"[</p>v/title]", "", dialoges[i])
                dialoges[i] = re.sub("\s\s+", "", dialoges[i])
                dialoges[i] = dialoges[i][1:]
                dialoges[i] = re.sub(r"(—)(.*)(—)", "", dialoges[i])
                dialoges[i] = re.sub(r"(—)(.*)", "", dialoges[i])
                dialoges[i] = re.sub(r"(–)(.*)(–)", "", dialoges[i])
                dialoges[i] = re.sub(r"([])(.*)(])", "", dialoges[i])
                dialoges[i] = re.sub(r"(–)(.*)", "", dialoges[i])
                dialoges[i] = re.sub(r"([.!?])(.*)", "", dialoges[i])
                if dialoges[i][-1] not in ["!","?","."]:
                    dialoges[i] = dialoges[i] + "."
                if dialoges[i][-2] in [".", ","]:
                    dialoges[i] = dialoges[i][:-2] + dialoges[i][-1]
                if dialoges[i][0] == " ":
                    dialoges[i] = dialoges[i][1:]
                dialoges[i] = re.sub("\s\s+", " ", dialoges[i])
            respond += dialoges[:-1]
            answer += dialoges[1:]
        return respond, answer
