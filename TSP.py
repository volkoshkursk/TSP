import numpy as np


def generate_task(num):
    out = np.random.sample((num, num))
    for i in range(num):
        out[i][i] = np.float('inf')
    return out


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

    def __lower_bound__(self, left):
        """
        нижняя гранца
        :param left: список городов, которые необходимо посетить
        :return:
        """
        left_matrix = np.array([[self.matrix[j][i] for i in left] for j in left])
        return sum(left_matrix.min(axis=1))

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
    #    t = TSP(generate_task(5), 5)
    a = generate_task(5)
    matrix = np.array([np.float('inf'), 10, 25, 25, 10,
                       1, np.float('inf'), 10, 15, 2,
                       8, 9, np.float('inf'), 20, 10,
                       14, 10, 24, np.float('inf'), 15,
                       10, 8, 25, 27, np.float('inf')])
    t = TSP(matrix.reshape((5, 5)), 5)
    print(t.__high_bound__([0, 1, 2, 3, 4]))
    print(t.__lower_bound__([0, 1, 2, 3, 4]))
