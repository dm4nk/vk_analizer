import re
from math import floor

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from num2words import num2words
from pymorphy2 import MorphAnalyzer
from tqdm import tqdm


class LikesDivider:
    def __init__(self, likes: [int], group_count: int):
        self.__group_count = group_count
        self.__likes_groups = {}
        self.__groups_likes = {}
        self.__likes = sorted(likes)
        self.__n = len(likes)
        self.__borders = []
        for i, like in tqdm(enumerate(self.__likes), desc="Generating likes groups"):
            current_gr = floor(i * group_count / self.__n)
            self.__likes_groups[like] = current_gr
            self.__groups_likes[current_gr] = like

        self.__borders.append(self.__likes[0])

        for i in range(1, self.__group_count):
            self.__borders.append(self.__likes[floor(i / self.__group_count * self.__n)])

        self.__borders.append(self.__likes[self.__n - 1])

        print('Likes groups: ', self.__likes_groups)
        print('Borders: ', self.__borders)

    def get_like_group(self, like: int):
        return self.__likes_groups[like]

    def get_group_borders(self, group: int):
        return self.__borders[group], self.__borders[group+1]

    def get_likes_groups(self):
        return [str(self.get_group_borders(g)) for g in range(self.__group_count)]


class Preprocessor:
    def __init__(self):
        self.__rt = RegexpTokenizer(r'\w+')
        self.__morph = MorphAnalyzer()
        self.__hashed_stop_words = [hash(word) for word in stopwords.words()]

    def __preprocess_text(self, tokens: str) -> str:
        tokens = self.__rt.tokenize(re.sub('http[s]?://\S+', '', tokens))
        text = ""

        for i, w in enumerate(tokens):
            tokens[i] = self.__morph.parse(num2words(w) if w.isnumeric() else w)[0].normal_form

        tokens = [words for words in tokens if not hash(words) in self.__hashed_stop_words]

        # i in range(1, len(tokens) - 1):
        #    text += tokens[i - 1] + tokens[i] + tokens[i + 1] + '\t'

        for w in tokens:
            text += w + " "

        return text

    def preprocess_texts(self, texts: [str]) -> [str]:
        processed_texts = []
        for text in tqdm(texts, desc="Preprocessing"):
            processed_texts.append(self.__preprocess_text(text))

        return processed_texts
