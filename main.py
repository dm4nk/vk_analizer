import numpy as np
import pymorphy2
from matplotlib import pyplot as plt
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR, NuSVR

from immaterial import data_from_skillbox_csv, draw_confusion_matrix
from transformers import PreprocessTransformer, LikesTransformer
from utils import VkApi

rt = RegexpTokenizer(r'\w+')
morph = pymorphy2.MorphAnalyzer()
tf_idf = TfidfVectorizer(ngram_range=(1, 2), max_features=150000)  # add ngram_range=(1, 2)
svr = NuSVR(verbose=True, kernel="linear", C=2000, gamma="auto")
preprocessor = PreprocessTransformer()
likes_transformer = LikesTransformer()
IMPORTANT_WORDS_COUNT = 10


# -*- coding: utf-8 -*-


def check_quality(texts, likes):
    texts = preprocessor.fit_transform(texts)
    texts = tf_idf.fit_transform(texts).toarray()

    train_texts, test_texts, train_likes, test_likes = train_test_split(texts, likes,
                                                                        test_size=0.66,
                                                                        shuffle=True)

    svr.fit(train_texts, train_likes)

    predicted_likes = svr.predict(test_texts)

    x = range(0, len(test_likes))
    plt.plot(x, test_likes, label='test_likes')
    plt.plot(x, predicted_likes, label='predicted_likes')
    plt.show()

    error = mean_squared_error(test_likes, predicted_likes)
    print('Mean quared_error: ' + str(error))

    test_likes_groups = likes_transformer.fit_transform(test_likes)
    predicted_likes_groups = [likes_transformer.get_like_group(like) for like in predicted_likes]

    print('Accuracy: ' + str(accuracy_score(test_likes_groups, predicted_likes_groups)))

    draw_confusion_matrix(predicted_likes_groups, test_likes_groups,
                          ['> ' + str(border) for border in likes_transformer.get_likes_groups()])


def get_most_important_words(transformed_text):
    feature_array = np.array(tf_idf.get_feature_names_out())
    tfidf_sorting = np.argsort(transformed_text).flatten()[::-1]
    return feature_array[tfidf_sorting][:IMPORTANT_WORDS_COUNT]


def execute(group_domain, text_to_estimate):
    texts, likes = VkApi('89277583192', 'password').get_group_posts_and_likes(group_domain)

    texts.append(text_to_estimate)

    texts = preprocessor.fit_transform(texts)
    texts_array = tf_idf.fit_transform(texts).toarray()
    svr.fit(texts_array[:-1], likes)

    predicted_like = svr.predict([texts_array[-1]])[0]

    most_important_words = get_most_important_words(texts_array)

    return predicted_like, most_important_words


def main():
    texts, likes = data_from_skillbox_csv()

    check_quality(texts[:1500], likes[:1500])


if __name__ == '__main__':
    main()
