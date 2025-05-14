from collections import deque


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
    network = FlowNetwork(4)
    # Нумерация вершин: 0 - s, 1 - a, 2 - b, 3 - t
    network.add_edge(0, 1, 3)  # s -> a
    network.add_edge(1, 3, 2)  # a -> t
    network.add_edge(0, 2, 2)  # s -> b
    network.add_edge(2, 3, 3)  # b -> t
    network.add_edge(1, 2, 1)  # a -> b

    s, t = 0, 3
    max_flow = network.max_flow(s, t)
    S, T = network.min_cut(s)

    print(f"Максимальный поток: {max_flow}")
    print(f"Минимальный разрез:")
    print(f"S = {S} (вершины, достижимые из истока в остаточной сети)")
    print(f"T = {T} (остальные вершины)")
    print(f"Пропускная способность разреза = {max_flow}")