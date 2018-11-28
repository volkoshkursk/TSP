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
        self.s_cost = 0
        self.discarded = []
        s = [i for i in range(num_of_towns)]
        s.append(0)
        self.__high_bound__(s)
        self.__lower_bound__(s)
        while not (self.__check()):
            self.__variants(self.__staff_matrix())
        m = min(self.discarded, key=lambda x: x[1])
        if not(m[1] > self.s_cost):
            self.matrix = copy(m[2])
            self.s = copy(m[0])
            self.s_cost = m[1]
            while not (self.__check()):
                self.__variants(self.__staff_matrix())
        self.__check_unique(self.matrix)

    def __score(self, matrix):
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

    def __generate_latent_ways(self, s=None):
        """
        костыльный метод вычисления подмаршрутов (в каких городах побывали)
        буду благодарен получить альтернативные решения этой проблемы
        :param s: новая точка маршрута
        :return: список всех подмаршрутов имеющегося маршрута
        """
        if s is None:
            s = copy(self.s)
        else:
            s = [s] + copy(self.s)
        for i in s:
            for j in s:
                if i[1] == j[0] and not ((i[1], j[0]) in set(s)) and i[1] != 0:
                    s.append((i[0], j[1]))
        s += list(map(lambda x: (x[1], x[0]), s))
        return s

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

    def __staff_matrix(self, matrix=None):
        """
        матрица С"
        список координат с нулями и разность
        """
        def check_inf(a): return 0 if a == np.float('inf') else a

        if matrix is None:
            new_matrix = copy(self.matrix)
        else:
            new_matrix = matrix

        m1 = new_matrix.min(axis=1)
        m1 = np.array(list(map(check_inf, m1)))
        new_matrix = (new_matrix.T - m1).T

        m0 = new_matrix.min(axis=0)
        m0 = np.array(list(map(check_inf, m0)))
        new_matrix = new_matrix - m0

        nulls = list(map(lambda x: (x[0], x[1]), np.argwhere(new_matrix == 0)))
        return new_matrix, nulls, sum(m0) + sum(m1)

    def __variants(self, arg):
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
        #        h1 = 0
        #        h2 = 0
        #        for i in range(self.num_of_towns):
        #            if new_matrix[i][now[0][1]] != np.float('inf'):
        #                h1 += new_matrix[i][now[0][1]]
        #            if new_matrix[now[0][0]][i] != np.float('inf'):
        #                h2 += new_matrix[now[0][0]][i]

        temp_m = copy(new_matrix)
        for i in range(self.num_of_towns):
            temp_m[i][now[0][1]] = np.float('inf')
            temp_m[now[0][0]][i] = np.float('inf')
        #        temp_m[now[0][0]][now[0][1]] = np.float('inf')
        temp_m[now[0][1]][now[0][0]] = np.float('inf')

        for i in self.__generate_latent_ways(now[0]):
            temp_m[i[0]][i[1]] = np.float('inf')
        temp_m, _, lower_with = self.__staff_matrix(temp_m)
        lower_with += minuses
        #        lower_with = minuses + self.__score(temp_m)

        for i in self.__generate_latent_ways():
            new_matrix[i[0]][i[1]] = np.float('inf')
        new_matrix[now[0][0]][now[0][1]] = np.float('inf')
        
        if (minuses + now[1]) > lower_with:
            self.s.append(copy(now[0]))
            self.matrix = copy(temp_m)
            self.discarded.append((copy(self.s), self.s_cost + minuses + now[1], copy(new_matrix)))
            self.s_cost += lower_with
            del temp_m, new_matrix, nulls, now
        else:
            self.discarded.append((copy(self.s) + [copy(now[0])], self.s_cost + lower_with, copy(temp_m)))
            del temp_m
            self.s_cost += minuses + now[1]
            self.matrix, _, _ = self.__staff_matrix(new_matrix)
            del new_matrix
            self.matrix[now[0][0]][now[0][1]] = np.float('inf')
            del now

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

    def __check_unique(self, matrix):
    	m0 = matrix.min(axis = 0)
    	if min(m0) == np.float("inf"):
    		print("Way is unique")

    def show(self):
        return str(self.s) + '\n' + str(self.s_cost)


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