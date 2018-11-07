import numpy as np


def generate_task(num):
    """
    создание матрицы расстояний
    :param num: количество городов
    :return:
    """
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
        self.s = []

    def __high_bound__(self, left):
        """
        верхняя гранца
        :param left: список городов, которые необходимо посетить
        :return:
        """
        s = [i for i in left]
        s.append(0)
        self.high = self.f(s)

    def __lower_bound__(self, left):
        """
        нижняя гранца
        :param left: список городов, которые необходимо посетить
        :return:
        """
        left_matrix = np.array([[self.matrix[j][i] for i in left] for j in left])
        self.lower = sum(left_matrix.min(axis=1))

    def __staff_matrix__(self):
        """
        матрица С"
        список координат с нулями и разность
        """
        self.new_matrix = self.matrix
        self.nulls = []
        m1 = self.new_matrix.min(axis=1)
        for i in range(self.num_of_towns):
            for j in range(self.num_of_towns):
                self.new_matrix[i][j] -= m1[i]
        m0 = self.new_matrix.min(axis=0)
        for i in range(self.num_of_towns):
            for j in range(self.num_of_towns):
                self.new_matrix[i][j] -= m0[j]
                if self.new_matrix[i][j] == 0:
                    self.nulls.append((i, j))
        self.minuses = sum(m0)+sum(m1)

    def __variants__(self):
        now = (None, 0)
        for i in self.nulls:
            res = min(np.concatenate((self.new_matrix[:i[0], i[1]],  self.new_matrix[i[0]+1:, i[1]]))) + \
                  min(np.concatenate((self.new_matrix[i[0], :i[1]], self.new_matrix[i[0], i[1]+1:])))
            if now[1] < res:
                now = (i, res)
        h1 = 0
        h2 = 0
        for i in range(self.num_of_towns):
            if self.new_matrix[i][now[0][1]] != np.float('inf'):
                h1 += self.new_matrix[i][now[0][1]]
            if self.new_matrix[now[0][0]][i] != np.float('inf'):
                h2 += self.new_matrix[now[0][0]][i]
        if self.lower < (self.minuses + now[1]):
            self.s.append(now[0])
            for i in range(self.num_of_towns):
                self.matrix[i][now[0][1]] = np.float('inf')
                self.matrix[now[0][0]][i] = np.float('inf')
            self.matrix[now[0][0]][now[0][1]] = np.float('inf')

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
    print(t.__staff_matrix__())
    t.__variants__()
    print()
