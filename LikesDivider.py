from math import floor


class LikesDivider:
    def __init__(self, likes: [int], group_count: int):
        self.__group_count = group_count

        self.__borders = []
        sorted_likes = sorted(likes)
        n = len(sorted_likes)

        for i in range(1, group_count + 1):
            right_border = sorted_likes[floor(i / group_count * (n - 1))]
            self.__borders.append(right_border)

    def get_like_group(self, like: int):
        i = 0
        while like > self.__borders[i]:
            i += 1
        return i
