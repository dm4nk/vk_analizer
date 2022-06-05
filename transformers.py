from sklearn.base import TransformerMixin, BaseEstimator

from utils import Preprocessor, LikesDivider

GROUP_COUNT = 3


class PreprocessTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.__preprocessor = Preprocessor()
        self.__X = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = self.__preprocessor.preprocess_texts(X)
        return X


class LikesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.__likes_divider = None

    def fit(self, X, y=None):
        self.__likes_divider = LikesDivider(X, GROUP_COUNT)
        return self

    def transform(self, X, y=None):
        return [self.__likes_divider.get_like_group(like) for like in X]

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    def get_likes_groups(self):
        return self.__likes_divider.get_likes_groups()

    def get_like_group(self, like: int):
        return self.__likes_divider.get_like_group(like)

    def get_group_right_border(self, group: int):
        return self.__likes_divider.get_group_right_border(group)
