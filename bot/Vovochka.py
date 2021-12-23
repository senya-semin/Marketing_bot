from nltk import text
import telebot

from Preparator import *
from Evgesha import *
from Pupsen import *
from Vupsen import *
from Seq2Seq import *

from numpy import random, argmax, amax, zeros
import pandas as pd

from telebot import types

preparator = Preparator()
preparator.init_ft()

talk = Seq2Seq(preparator)
evgesha = Evgesha(preparator)
pupsen = Pupsen(preparator)
vupsen = Vupsen(preparator)

evgesha.learn()
pupsen.learn()
vupsen.learn()

emotions = {
    0: "радость",
    1: "удивление",
    2: "грусть",
    3: "страх",
    4: "гнев",
    5: "отвращение",
}

themes = [
    "кино",
    "мультик",
    "аниме",
    "политика",
    "спорт",
    "новость",
    "любовь",
    "гнев",
    "страх",
    "отвращение",
    "грусть",
    "радость",
    "удивление",
    "игры",
    "учеба",
    "работа",
    "деньги",
    "успех",
    "неудачи",
    "сериалы",
    "искусство",
    "развлечение",
    "европа",
    "жуть",
    "гадость",
    "мерзость",
    "шок",
    "чудо",
    "ярость",
]

bot = telebot.TeleBot("")


def get_markup():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    item1 = types.InlineKeyboardButton("Получить сообщение")
    item2 = types.InlineKeyboardButton("Ввести сообщение")
    item3 = types.InlineKeyboardButton("Поговорить")
    markup.add(item1, item2, item3)
    return markup


def add_information(messedge, text, keyboard):
    preparator.answer = preparator.clear_text(text)
    importance = pupsen.predict(text)
    if importance[0] <= 0.5:
        preparator.importance = 0
        preparator.emotions = evgesha.predict(text)
        bot.send_message(
            messedge.chat.id,
            "Я не вижу в этом тектсе ничего важного, но я чувствую {}".format(
                emotions[argmax(preparator.emotions)]
            ),
            reply_markup=keyboard,
        )
    else:
        emotion = evgesha.predict(text)
        preparator.emotions = emotion
        emotion_main = emotions[argmax(emotion)]

        plot, proba = vupsen.predict(text)
        preparator.plot = plot
        preparator.proba = proba
        preparator.importance = 1

        bot.send_message(
            messedge.chat.id,
            "Мне кажется, что автор испытывает {} из-за {}".format(emotion_main, plot),
            reply_markup=keyboard,
        )


@bot.message_handler(commands=["start"])
def start_message(message):

    markup = get_markup()

    bot.send_message(
        message.chat.id,
        "Привет! \n"
        + "Я Вовчка! Рад с тобой познакомиться \n"
        + "Ты можешь отправить мне сообщение, а я попытаюсь понять о чем оно или я могу сам сгенерировать сообщение, а ты меня проверишь.\n"
        + "Или же ты можешь попытаться со мной поговрить, но имеей ввиду, что я ещё очень глупый\n"
        + "Давай сыграем, будет весело!))",
        reply_markup=markup,
    )


@bot.message_handler(content_types=["text"])
def act(messedge):
    if messedge.chat.type == "private":
        if messedge.text == "Получить сообщение":
            keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
            item_concrete = types.InlineKeyboardButton("Конкретную")
            item_random = types.InlineKeyboardButton("Все равно")
            keyboard.add(item_concrete, item_random)
            bot.send_message(
                messedge.chat.id,
                "Ты хочешь получить сообщение на какую-то конкретную тему или тебе все равно?",
                reply_markup=keyboard,
            )
            bot.register_next_step_handler(messedge, ways)

        elif messedge.text == "Ввести сообщение":
            bot.send_message(
                messedge.chat.id, "Введи сообщение и посмотрим, что я смогу сказать"
            )
            bot.register_next_step_handler(messedge, read)

        elif messedge.text == "Поговорить":
            bot.send_message(
                messedge.chat.id, "Да начнется диалог!"
            )
            bot.register_next_step_handler(messedge, speak)

def speak(messedge):
    text= messedge.text
    print(text)
    answer = talk.predict(text)
    bot.send_message(
                messedge.chat.id, answer
            )
    bot.register_next_step_handler(messedge, speak)

def ways(messedge):
    if messedge.text == "Конкретную":
        bot.send_message(messedge.chat.id, "Хорошо, отправь мне тему")
        bot.register_next_step_handler(messedge, get_messedge)

    elif messedge.text == "Все равно":
        keyboard = types.InlineKeyboardMarkup()
        key_yes = types.InlineKeyboardButton(
            text="Я тоже", callback_data="yes"
        )  # он ответил правильно
        key_no = types.InlineKeyboardButton(
            text="Не согласен", callback_data="no"
        )  # он ответил неправильно
        key_details = types.InlineKeyboardButton(
            text="Почему так?", callback_data="details"
        )  # уточнить вероятность эмоции
        key_pass = types.InlineKeyboardButton(
            text="Пропустить", callback_data="pass"
        )  # уточнить вероятность эмоции
        keyboard.add(key_yes, key_no, key_details, key_pass)

        bot.send_message(messedge.chat.id, "Думаю, что сказать, подожди")
        theme = random.choice(themes)
        text: str = preparator.get_random(theme)  # дает рандомный твит
        bot.send_message(messedge.chat.id, text)

        add_information(messedge, text, keyboard)


def get_messedge(messedge):

    keyboard = types.InlineKeyboardMarkup()
    key_yes = types.InlineKeyboardButton(
        text="Я тоже", callback_data="yes"
    )  # он ответил правильно
    key_no = types.InlineKeyboardButton(
        text="Не согласен", callback_data="no"
    )  # он ответил неправильно
    key_details = types.InlineKeyboardButton(
        text="Почему так?", callback_data="details"
    )  # уточнить вероятность эмоции
    key_pass = types.InlineKeyboardButton(
        text="Пропустить", callback_data="pass"
    )  # уточнить вероятность эмоции
    keyboard.add(key_yes, key_no, key_details, key_pass)

    bot.send_message(messedge.chat.id, "Думаю, что сказать, подожди")
    text = messedge.text
    answer: str = preparator.get_random(text)  # дает рандомный твит
    bot.send_message(messedge.chat.id, answer)

    add_information(messedge, text, keyboard)


def read(messedge):

    keyboard = types.InlineKeyboardMarkup()
    key_yes = types.InlineKeyboardButton(
        text="Я тоже", callback_data="yes"
    )  # он ответил правильно
    key_no = types.InlineKeyboardButton(
        text="Не согласен", callback_data="no"
    )  # он ответил неправильно
    key_details = types.InlineKeyboardButton(
        text="Почему так?", callback_data="details"
    )  # уточнить вероятность эмоции
    key_pass = types.InlineKeyboardButton(
        text="Пропустить", callback_data="pass"
    )  # уточнить вероятность эмоции
    keyboard.add(key_yes, key_no, key_details, key_pass)

    text = messedge.text

    add_information(messedge, text, keyboard)


def difword_(messedge):
    bot.send_message(messedge.chat.id, "Хорошо, отправь мне ключевое слово")
    bot.register_next_step_handler(messedge, difword)


def difword(messedge):
    plot = messedge.text
    print(plot)
    preparator.plot = plot

    keyboard = types.InlineKeyboardMarkup()
    emotion_step = types.InlineKeyboardButton(text="С эмоцией", callback_data="emotion")
    end = types.InlineKeyboardButton(text="Нет", callback_data="yes")

    keyboard.add(emotion_step, end)
    importance = 0

    bot.send_message(messedge.chat.id, "У меня есть ещё ошибки?", reply_markup=keyboard)


@bot.callback_query_handler(func=lambda call: True)
def bot_answer(call):
    answer = preparator.answer
    emotion = preparator.emotions
    plot = preparator.plot
    proba = preparator.proba
    importance = preparator.importance

    if call.data == "yes":

        data = pd.DataFrame([answer, importance, plot])
        data.to_csv("текст.csv", mode="a", header=False)

        data = pd.DataFrame([answer, argmax(emotion[0])])
        data.to_csv("Эмоции.csv", mode="a", header=False)

        markup = get_markup()
        bot.send_message(
            call.message.chat.id, "Спасибо) \n" + "Продолжим игру?", reply_markup=markup
        )

    elif call.data == "no":
        keyboard = types.InlineKeyboardMarkup()
        emotion_step = types.InlineKeyboardButton(
            text="Эмоцией", callback_data="emotion"
        )
        difword = types.InlineKeyboardButton(
            text="Ключевым словом", callback_data="difword"
        )
        noword = types.InlineKeyboardButton(text="Важностью", callback_data="noword")
        keyboard.add(emotion_step, difword, noword)
        bot.send_message(
            call.message.chat.id,
            "Ты не согласен с эмоцией или ключевым словом?",
            reply_markup=keyboard,
        )

    elif call.data == "details":
        text = f"В своих чувствах я уверен на {amax(preparator.emotions):.2f}%, а в правильность слова на {proba:.2f}%"
        text = text[:-2]
        keyboard = types.InlineKeyboardMarkup()
        key_yes = types.InlineKeyboardButton(
            text="Я тоже", callback_data="yes"
        )  # он ответил правильно
        key_no = types.InlineKeyboardButton(
            text="Не согласен", callback_data="no"
        )  # он ответил неправильно
        keyboard.add(key_yes, key_no)
        bot.send_message(call.message.chat.id, text, reply_markup=keyboard)

    elif call.data == "pass":
        markup = get_markup()
        bot.send_message(
            call.message.chat.id,
            "Видимо, что-то пошло не так... \n" + "Продолжим игру?",
            reply_markup=markup,
        )

    elif call.data == "noword":
        if preparator.importance == 0:
            emotion = evgesha.predict(answer)
            preparator.emotions = emotion
            emotion_main = emotions[argmax(emotion)]

            plot, proba = vupsen.predict(answer)
            preparator.plot = plot
            preparator.proba = proba
            preparator.importance = 1
            bot.send_message(
                call.message.chat.id,
                "Мне кажется, что автор испытывает {} из-за {}".format(
                    emotion_main, plot
                ),
            )

        else:
            importance = 0
            preparator.importance = importance
            preparator.plot = ""

        keyboard = types.InlineKeyboardMarkup()
        no = types.InlineKeyboardButton(text="нет", callback_data="no")
        end = types.InlineKeyboardButton(text="да", callback_data="yes")

        keyboard.add(no, end)

        bot.send_message(
            call.message.chat.id, "Тут все правильно?", reply_markup=keyboard
        )

    elif call.data == "difword":
        difword_(call.message)

    elif call.data == "emotion":
        keyboard = types.InlineKeyboardMarkup()
        happines = types.InlineKeyboardButton(text="Радость", callback_data="0")
        wandering = types.InlineKeyboardButton(text="Удивление", callback_data="1")
        sadness = types.InlineKeyboardButton(text="Грусть", callback_data="2")
        fear = types.InlineKeyboardButton(text="Страх", callback_data="3")
        angry = types.InlineKeyboardButton(text="Гнев", callback_data="4")
        disgust = types.InlineKeyboardButton(text="Отвращение", callback_data="5")
        keyboard.add(happines, wandering, sadness, fear, angry, disgust)
        bot.send_message(
            call.message.chat.id, "Что же ты чувствуешь?", reply_markup=keyboard
        )

    else:
        emotion = [zeros(6)]
        emotion[0][int(call.data)] = 1
        preparator.emotions = emotion
        keyboard = types.InlineKeyboardMarkup()
        plot_step = types.InlineKeyboardButton(
            text="С ключевым словом", callback_data="difword"
        )
        end = types.InlineKeyboardButton(
            text="С остальным я согласен", callback_data="yes"
        )
        keyboard.add(plot_step, end)
        bot.send_message(
            call.message.chat.id, "Спасибо, есть ещё проблемы", reply_markup=keyboard
        )


bot.polling(none_stop=True)
