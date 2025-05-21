from collections import deque
from random import randint

class Edge:
    def __init__(self, to, rev, capacity):
        self.to = to  # Куда ведёт ребро
        self.rev = rev  # Обратное ребро
        self.capacity = capacity  # Пропускная способность
        self.flow = 0  # Текущий поток


class FlowNetwork:
    def __init__(self, n):
        self.size = n
        self.graph = [[] for _ in range(n)]  # Список смежности

    def add_edge(self, fr, to, capacity):
        # Прямое ребро
        forward = Edge(to, len(self.graph[to]), capacity)
        # Обратное ребро (для остаточной сети)
        backward = Edge(fr, len(self.graph[fr]), 0)
        self.graph[fr].append(forward)
        self.graph[to].append(backward)

    def bfs_level_graph(self, s, t, level):
        queue = deque()
        queue.append(s)
        level[s] = 1

        while queue:
            v = queue.popleft()
            for edge in self.graph[v]:
                if level[edge.to] == 0 and edge.flow < edge.capacity:
                    level[edge.to] = level[v] + 1
                    queue.append(edge.to)

        return level[t] != 0

    def dfs_send_flow(self, v, t, flow, level, ptr):
        if v == t:
            return flow

        for i in range(ptr[v], len(self.graph[v])):
            edge = self.graph[v][i]
            if level[edge.to] == level[v] + 1 and edge.flow < edge.capacity:
                min_flow = min(flow, edge.capacity - edge.flow)
                result = self.dfs_send_flow(edge.to, t, min_flow, level, ptr)

                if result > 0:
                    edge.flow += result
                    self.graph[edge.to][edge.rev].flow -= result
                    return result

            ptr[v] += 1

        return 0

    def max_flow(self, s, t):
        max_flow = 0
        level = [0] * self.size

        while self.bfs_level_graph(s, t, level):
            ptr = [0] * self.size
            while True:
                flow = self.dfs_send_flow(s, t, float('inf'), level, ptr)
                if flow == 0:
                    break
                max_flow += flow

            level = [0] * self.size

        return max_flow

    def min_cut(self, s):
        visited = [False] * self.size
        queue = deque()
        queue.append(s)
        visited[s] = True

        while queue:
            v = queue.popleft()
            for edge in self.graph[v]:
                if not visited[edge.to] and edge.flow < edge.capacity:
                    visited[edge.to] = True
                    queue.append(edge.to)

        # Вершины, достижимые из s в остаточной сети
        S = [i for i in range(self.size) if visited[i]]
        T = [i for i in range(self.size) if not visited[i]]
        return S, T


# Пример использования
if __name__ == "__main__":
    # Создаём сеть из примера
    network = FlowNetwork(8)
    # Нумерация вершин: 0 - s, 1 - a, 2 - b, 3 - c, 4 - d, 5 - p, 6 - k, 7 - t
    network.add_edge(0, 1, 13)  # s -> a
    network.add_edge(0, 4, 15)  # s -> d
    network.add_edge(0, 5, 7)  # s -> p
    network.add_edge(1, 4, 12)  # a -> d
    network.add_edge(1, 6, 11)  # a -> k
    network.add_edge(1, 2, 6)  # a -> b
    network.add_edge(2, 7, 5)  # b -> t
    network.add_edge(3, 2, 11)  # c -> b
    network.add_edge(3, 7, 14)  # c -> t
    network.add_edge(4, 3, 5)  # d -> c
    network.add_edge(5, 2, 13)  # p -> b
    network.add_edge(5, 6, 12)  # p -> k
    network.add_edge(6, 7, 6)  # k -> t

    s, t = 0, 7
    max_flow = network.max_flow(s, t)
    S, T = network.min_cut(s)
    print("Пример из лабораторной")
    print(f"Максимальный поток: {max_flow}")
    print(f"Минимальный разрез:")
    print(f"S = {S} (вершины, достижимые из истока в остаточной сети)")
    print(f"T = {T} (остальные вершины)")
    print(f"Пропускная способность разреза = {max_flow}")
    print("\n \n \n")


    #Случайные значения
    network1 = FlowNetwork(8)
    # Нумерация вершин: 0 - s, 1 - a, 2 - b, 3 - c, 4 - d, 5 - p, 6 - k, 7 - t
    network1.add_edge(0, 1, randint(100, 1000))  # s -> a
    network1.add_edge(0, 4, randint(100, 1000))  # s -> d
    network1.add_edge(0, 5, randint(100, 1000))  # s -> p
    network1.add_edge(1, 4, randint(100, 1000))  # a -> d
    network1.add_edge(1, 6, randint(100, 1000))  # a -> k
    network1.add_edge(1, 2, randint(100, 1000))  # a -> b
    network1.add_edge(2, 7, randint(100, 1000))  # b -> t
    network1.add_edge(3, 2, randint(100, 1000))  # c -> b
    network1.add_edge(3, 7, randint(100, 1000))  # c -> t
    network1.add_edge(4, 3, randint(100, 1000))  # d -> c
    network1.add_edge(5, 2, randint(100, 1000))  # p -> b
    network1.add_edge(5, 6, randint(100, 1000))  # p -> k
    network1.add_edge(6, 7, randint(100, 1000))  # k -> t

    s, t = 0, 7
    max_flow = network1.max_flow(s, t)
    S, T = network1.min_cut(s)

    print("Пример со случайными значениями")
    print(f"Максимальный поток: {max_flow}")
    print(f"Минимальный разрез:")
    print(f"S = {S} (вершины, достижимые из истока в остаточной сети)")
    print(f"T = {T} (остальные вершины)")
    print(f"Пропускная способность разреза = {max_flow}")
