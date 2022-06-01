import numpy as np
import pymorphy2
import vk_api
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

from immaterial import data_from_skillbox_csv
from transformers import PreprocessTransformer

rt = RegexpTokenizer(r'\w+')
morph = pymorphy2.MorphAnalyzer()
tf_idf = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, max_features=150000)

nu_svc = SVR(verbose=True, max_iter=-1, cache_size=1000)
preprocessor = PreprocessTransformer()
hashed_stop_words = [hash(word) for word in stopwords.words()]
IMPORTANT_WORDS_COUNT = 10


# -*- coding: utf-8 -*-


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

    texts, likes = [], []

    for i in range(0, 5):
        json_data = vk.wall.get(domain=group_domain, count=100, filter='owner', offset=100 * i)
        print('Got ' + str(i) + ' response')
        t, l = get_texts_and_likes_from_json(json_data)
        texts += t
        likes += l

    print(texts)
    return texts, likes


def check_quality(texts, likes):
    texts = preprocessor.fit_transform(texts)
    texts = tf_idf.fit_transform(texts).toarray()

    train_texts, test_texts, train_likes, test_likes = train_test_split(texts, likes,
                                                                        test_size=0.1,
                                                                        shuffle=True)

    nu_svc.fit(train_texts, train_likes)

    predicted_likes = nu_svc.predict(test_texts)

    print(test_likes)
    print(predicted_likes)
    x = range(0, len(test_likes))

    plt.plot(x, test_likes, label='test_likes')
    plt.plot(x, predicted_likes, label='predicted_likes')

    errors = mean_squared_error(test_likes, predicted_likes)
    print(errors)
    plt.show()


def get_most_important_words(transformed_text):
    feature_array = np.array(tf_idf.get_feature_names_out())
    tfidf_sorting = np.argsort(transformed_text).flatten()[::-1]
    return feature_array[tfidf_sorting][:IMPORTANT_WORDS_COUNT]


def execute(group_domain, text_to_estimate):
    texts, likes = get_group_posts_and_likes("89277583192", "", group_domain)

    texts.append(text_to_estimate)

    texts = preprocessor.fit_transform(texts)
    texts_array = tf_idf.fit_transform(texts).toarray()
    nu_svc.fit(texts_array[:-1], likes)

    predicted_like = nu_svc.predict([texts_array[-1]])[0]

    most_important_words = get_most_important_words(texts_array)

    return predicted_like, most_important_words


def main():
    texts, likes = data_from_skillbox_csv()
    # texts, likes = get_test_data_from_file('data/data_from_javatutorial.json')

    check_quality(texts, likes)

    # borders, most_important_words = execute('psyhology', get_text())
    # print('Predicted Like: ' + str(borders))
    # print('Words: ' + str(most_important_words))


if __name__ == '__main__':
    main()
