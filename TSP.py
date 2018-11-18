import numpy as np
from copy import copy


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
        self.discarded = []
        s = [i for i in range(num_of_towns)]
        s.append(0)
        self.__high_bound__(s)
        self.__lower_bound__(s)
        while not (self.__check()):
            self.__variants__(self.__staff_matrix())

    def __score__(self, matrix):
        """
        оценка "вычетов" матрицы
        """
        m1 = matrix.min(axis=1)
        m0 = matrix.min(axis=0)
        for i in range(self.num_of_towns):
            if m0[i] == np.float('inf'):
                m0[i] = 0
            if m1[i] == np.float('inf'):
                m1[i] = 0
        return sum(m0) + sum(m1)

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

    def __staff_matrix(self):
        """
        матрица С"
        список координат с нулями и разность
        """
        new_matrix = self.matrix
        nulls = []
        m1 = new_matrix.min(axis=1)
        for i in range(self.num_of_towns):
            if m1[i] == np.float('inf'):
                m1[i] = 0
        for i in range(self.num_of_towns):
            for j in range(self.num_of_towns):
                new_matrix[i][j] -= m1[i]
        m0 = new_matrix.min(axis=0)
        for i in range(self.num_of_towns):
            if m0[i] == np.float('inf'):
                m0[i] = 0
        for i in range(self.num_of_towns):
            for j in range(self.num_of_towns):
                new_matrix[i][j] -= m0[j]
                if new_matrix[i][j] == 0:
                    nulls.append((i, j))
        return new_matrix, nulls, sum(m0) + sum(m1)

    def __variants__(self, arg):
        """
        основная функция
        :param arg: функция __staff_matrix
        :return: None
        """
        new_matrix, nulls, minuses = arg
        now = (None, 0)
        for i in nulls:
            res = min(np.concatenate((new_matrix[:i[0], i[1]], new_matrix[i[0] + 1:, i[1]]))) + \
                  min(np.concatenate((new_matrix[i[0], :i[1]], new_matrix[i[0], i[1] + 1:])))
            if now[1] < res:
                now = (i, res)
        h1 = 0
        h2 = 0
        for i in range(self.num_of_towns):
            if new_matrix[i][now[0][1]] != np.float('inf'):
                h1 += new_matrix[i][now[0][1]]
            if new_matrix[now[0][0]][i] != np.float('inf'):
                h2 += new_matrix[now[0][0]][i]
        temp_m = copy(new_matrix)
        for i in range(self.num_of_towns):
            temp_m[i][now[0][1]] = np.float('inf')
            temp_m[now[0][0]][i] = np.float('inf')
        #        temp_m[now[0][0]][now[0][1]] = np.float('inf')
        temp_m[now[0][1]][now[0][0]] = np.float('inf')
        lower_with = minuses + self.__score__(temp_m)
        if (minuses + now[1]) > lower_with:
            self.s.append(now[0])
            self.lower = lower_with
            self.matrix = copy(temp_m)
            self.discarded.append((self.s + [None], minuses + now[1]))
            del temp_m, new_matrix, nulls
        else:
            del temp_m
            self.discarded.append((self.s + [now[0]], lower_with))
            self.matrix = copy(new_matrix)
            del new_matrix
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

    def __check(self):
        """
        проверка маршрута
        :return: маршрут полный или нет
        """
        si = set()
        so = set()
        for i in self.s:
            si.add(i[0])
            so.add(i[1])
        return len(si) == self.num_of_towns and len(so) == self.num_of_towns

    def show(self):
        return self.s


if __name__ == '__main__':
    #    t = TSP(generate_task(5), 5)
    #    a = generate_task(5)
    input_matrix = np.array([np.float('inf'), 10, 25, 25, 10,
                             1, np.float('inf'), 10, 15, 2,
                             8, 9, np.float('inf'), 20, 10,
                             14, 10, 24, np.float('inf'), 15,
                             10, 8, 25, 27, np.float('inf')])
    t = TSP(input_matrix.reshape((5, 5)), 5)
    print(t.show())
