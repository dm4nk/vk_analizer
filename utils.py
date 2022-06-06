import re

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from num2words import num2words
from pymorphy2 import MorphAnalyzer
from tqdm import tqdm
import vk_api


class LikesDivider:
    def __init__(self, likes: [int], group_count: int):
        self.__group_count = group_count
        self.__likes = sorted(likes)
        likes_len = len(likes)
        self.__borders = []
        if self.__group_count > likes_len:
            raise AttributeError('group_count > len(likes)')

        step = int(likes_len / group_count)

        for i in range(group_count):
            self.__borders.append(self.__likes[i * step])

        print('Borders: ' + str(self.__borders))

    def get_like_group(self, like: int):
        if like <= self.__borders[0]:
            return 0

        for i in range(len(self.__borders) - 1):
            if self.__borders[i] < like <= self.__borders[i + 1]:
                return i

        if like > self.__borders[-1]:
            return self.__group_count - 1

        return None

    def get_group_right_border(self, group: int):
        return self.__borders[group]

    def get_likes_groups(self):
        return [str(self.get_group_right_border(g)) for g in range(self.__group_count)]


class Preprocessor:
    def __init__(self):
        self.__rt = RegexpTokenizer(r'\w+')
        self.__morph = MorphAnalyzer()
        self.__hashed_stop_words = set([hash(word) for word in stopwords.words()])

    def __preprocess_text(self, tokens: str) -> str:
        tokens = self.__rt.tokenize(re.sub('http[s]?://\S+', '', tokens))
        text = ""

        for i, w in enumerate(tokens):
            tokens[i] = self.__morph.parse(num2words(w) if w.isdecimal() else w)[0].normal_form

        tokens = [words for words in tokens if not hash(words) in self.__hashed_stop_words]

        for w in tokens:
            text += w + " "

        return text

    def preprocess_texts(self, texts: [str]) -> [str]:
        processed_texts = []
        for text in tqdm(texts, desc="Preprocessing"):
            processed_texts.append(self.__preprocess_text(text))

        return processed_texts


def get_texts_and_likes_from_json(json_data):
    texts = []
    likes = []
    for item in json_data['items']:
        likes.append(item['likes']['count'])
        texts.append(item['text'])

    return texts, likes


def auth_handler():
    return input("Enter authentication code: "), True


class VkApi:
    def __init__(self, login, password):
        self.__login = login
        self.__password = password

    def get_group_posts_and_likes(self, group_domain) -> (str, str):
        vk_session = vk_api.VkApi(
            self.__login, self.__password,
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
