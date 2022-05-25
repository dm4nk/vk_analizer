from math import floor
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
