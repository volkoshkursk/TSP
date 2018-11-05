import numpy as np

# from random import random


def generate_task(num):
    return np.random.sample((num, num))


class TSP:
    """
    класс задача коммивояжера
    """
    def __init__(self, matrix, num_of_towns):
        self.matrix = matrix
        self.num_of_towns = num_of_towns

    def __high_bound__(self, left):
        """
        верхняя гранца
        :param left: список городов, которые необходимо посетить
        :return:
        """
        s = [i for i in left]
        s.append(0)
        return self.f(s)

    def f(self, s):
        """
        целевая функция
        :return:int "стоимость" пути
        """
        out = 0
        for i in range(1, len(s)):
            out += self.matrix[s[i - 1]][s[i]]
        return out


if __name__ == '__main__':
    generate_task(3)

