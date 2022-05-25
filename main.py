import json
import re

import pandas as pd
import pymorphy2
import vk_api
import xlsxwriter
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from num2words import num2words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import NuSVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
from tqdm import tqdm

from LikesDivider import LikesDivider

rt = RegexpTokenizer(r'\w+')
morph = pymorphy2.MorphAnalyzer()
tf_idf = TfidfVectorizer()
nu_svc = NuSVC(verbose=True, kernel='linear', max_iter=-1)
hashed_stop_words = [hash(word) for word in stopwords.words()]

# -*- coding: utf-8 -*-


def preprocess_text(tokens: str) -> str:
    tokens = rt.tokenize(re.sub('http[s]?://\S+', '', tokens))
    text = ""

    for i, w in enumerate(tokens):
        tokens[i] = morph.parse(num2words(w) if w.isnumeric() else w)[0].normal_form

    tokens = [words for words in tokens if not hash(words) in hashed_stop_words]

    # i in range(1, len(tokens) - 1):
    #    text += tokens[i - 1] + tokens[i] + tokens[i + 1] + '\t'

    for w in tokens:
        text += w + " "

    return text


def preprocess_texts(texts: [str]) -> [str]:
    processed_texts = []
    for text in tqdm(texts, desc="Preprocessing"):
        processed_texts.append(preprocess_text(text))

    return processed_texts


def tf_idf_matrix_to_exel(terms, tf_idf_matrix):
    workbook = xlsxwriter.Workbook('matrix.xlsx')
    worksheet = workbook.add_worksheet('sheet')
    row = 0
    col = 0
    for word in tqdm(terms, desc='Writing to exel'):
        worksheet.write(row, col, word)
        col += 1
    for post in tf_idf_matrix:
        row += 1
        col = 0
        for word in post:
            worksheet.write(row, col, word)
            col += 1
    workbook.close()


def auth_handler():
    return input("Enter authentication code: "), True


def get_texts_and_likes_from_json(json_data):
    texts = []
    likes = []
    for item in json_data['items']:
        likes.append(item['likes']['count'])
        texts.append(item['text'])

    return texts, likes


def get_group_posts_and_likes(login, password, group_domain) -> (str, str):
    vk_session = vk_api.VkApi(
        login, password,
        auth_handler=auth_handler
    )

    vk_session.auth()

    vk = vk_session.get_api()

    json_data = vk.wall.get(domain=group_domain)
    print("Response from vk:\n", json_data)

    return get_texts_and_likes_from_json(json_data)


def get_test_data_from_file(filename) -> (str, str):
    """
    Used for demonstration on test json
    :return:
    """
    with open(filename, "r", encoding='utf-8') as json_file:
        json_load = json.load(json_file)
    return get_texts_and_likes_from_json(json_load)


def split_data(x: [], y: [], test_size=1):
    if len(x) != len(y):
        raise AttributeError("x and y sizes does not match")
    if len(x) <= test_size:
        raise AttributeError("Too small test size")

    return x[test_size:], x[:test_size], y[test_size:], y[:test_size]


def get_text():
    return 'Прими участие в масштабном онлайн-хакатоне «Moscow City Hack 2022» от Агентства инноваций Москвы!Тебя ' \
           'ждут:🏆 Призовой фонд 3 400 000 ₽;💻 Задачи по разработке сервисов для мотивации студентов, привлечения ' \
           'волонтеров, цифрового маркетинга, импортозамещения и разоблачения fake news;🧐 Эксперты от крупного ' \
           'бизнеса и Правительства Москвы;🎓 Образовательная программа с мастер-классами и интенсивами;🎁 Красочный ' \
           'мерч, подарки от партнеров и много крутых активностей 🔥📅 Хакатон пройдет 10-13 июня 2022 ' \
           'годаРегистрируйся уже сейчас!https://bit.ly/3lhAjepУзнать подробности и найти команду можно в нашем ' \
           'Telegram-чатеt.me/MoscowCityHack '


def data_from_skillbox_csv():
    df = pd.read_csv('data/vk_skillbox.csv')

    return df['text'].apply(str).tolist(), df['likes'].tolist()


def check_quality(texts, likes):
    texts = preprocess_texts(texts)
    likes_divider = LikesDivider(likes, 3)
    likes_groups = [likes_divider.get_like_group(like) for like in likes]
    print(likes_groups)

    test_size = 100
    train_texts, test_texts, train_likes_groups, test_likes_groups = train_test_split(texts, likes_groups,
                                                                                      test_size=test_size,
                                                                                      shuffle=False)

    tf_idf_matrix = tf_idf.fit_transform(test_texts + train_texts).toarray()
    terms = tf_idf.get_feature_names_out()
    tf_idf_matrix_to_exel(terms, tf_idf_matrix)
    nu_svc.fit(tf_idf_matrix[test_size:], train_likes_groups)

    predicted_likes_groups = nu_svc.predict(tf_idf_matrix[:test_size])

    cm = confusion_matrix(test_likes_groups, predicted_likes_groups)

    df_cm = pd.DataFrame(cm, likes_divider.get_likes_groups(), likes_divider.get_likes_groups())
    sn.heatmap(df_cm, annot=True)
    plt.show()


def main():
    texts, likes = data_from_skillbox_csv()
    # texts, likes = get_test_data_from_file('data/data_from_javatutorial.json')
    check_quality(texts, likes)


if __name__ == '__main__':
    main()
