from collections import deque
from random import randint

class Edge:
    def __init__(self, to, rev, capacity):
        self.to = to
        self.rev = rev
        self.capacity = capacity
        self.flow = 0

class FlowNetwork:
    def __init__(self, n):
        self.size = n
        self.graph = [[] for _ in range(n)]

    def add_edge(self, fr, to, capacity):
        forward = Edge(to, len(self.graph[to]), capacity)
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

        S = [i for i in range(self.size) if visited[i]]
        T = [i for i in range(self.size) if not visited[i]]
        return S, T

    def verify_min_cut(self, max_flow_value, S, T):
        print("\n=== Проверка минимального разреза ===")
        print(f"Множество S: {S}")
        print(f"Множество T: {T}")

        edges_in_cut = []
        total_capacity = 0

        for u in S:
            for edge in self.graph[u]:
                if edge.to in T and edge.capacity > 0:  # Добавляем проверку capacity > 0
                    edges_in_cut.append((u, edge.to, edge.capacity))
                    total_capacity += edge.capacity

        print("\nРёбра разреза (S -> T):")
        for u, v, cap in edges_in_cut:
            print(f"{u} -> {v} (пропускная способность: {cap})")

        print(f"\nСуммарная пропускная способность разреза: {total_capacity}")
        print(f"Максимальный поток: {max_flow_value}")

        if total_capacity == max_flow_value:
            print("\n✅ Проверка пройдена: сумма пропускных способностей равна максимальному потоку.")
            print("Разрез корректен и является минимальным.")
        else:
            print("\n❌ Ошибка: сумма пропускных способностей НЕ равна максимальному потоку!")
            print("Возможные причины:")
            print("- Неправильно определён разрез (вершины S/T)")
            print("- Ошибка в вычислении максимального потока")
            print("- Пропущены некоторые рёбра разреза")

        return total_capacity == max_flow_value

if __name__ == "__main__":
    # Пример из лабораторной
    # Нумерация вершин: 0 - s, 1 - a, 2 - b, 3 - c, 4 - d, 5 - p, 6 - k, 7 - t
    network = FlowNetwork(8)
    network.add_edge(0, 1, 13)  # s -> a
    network.add_edge(0, 4, 15)  # s -> d
    network.add_edge(0, 5, 7)   # s -> p
    network.add_edge(1, 4, 12)  # a -> d
    network.add_edge(1, 6, 11)  # a -> k
    network.add_edge(1, 2, 6)   # a -> b
    network.add_edge(2, 7, 5)   # b -> t
    network.add_edge(3, 2, 11)  # c -> b
    network.add_edge(3, 7, 14)  # c -> t
    network.add_edge(4, 3, 5)   # d -> c
    network.add_edge(5, 2, 13)  # p -> b
    network.add_edge(5, 6, 12)  # p -> k
    network.add_edge(6, 7, 6)   # k -> t

    s, t = 0, 7
    max_flow = network.max_flow(s, t)
    S, T = network.min_cut(s)
    print("Пример из лабораторной")
    print(f"Максимальный поток: {max_flow}")
    print(f"Минимальный разрез:")
    print(f"S = {S}")
    print(f"T = {T}")
    network.verify_min_cut(max_flow, S, T)

    # Случайные значения
    network1 = FlowNetwork(8)
    # Нумерация вершин: 0 - s, 1 - a, 2 - b, 3 - c, 4 - d, 5 - p, 6 - k, 7 - t
    network1.add_edge(0, 1, randint(100, 1000))
    network1.add_edge(0, 4, randint(100, 1000))
    network1.add_edge(0, 5, randint(100, 1000))
    network1.add_edge(1, 4, randint(100, 1000))
    network1.add_edge(1, 6, randint(100, 1000))
    network1.add_edge(1, 2, randint(100, 1000))
    network1.add_edge(2, 7, randint(100, 1000))
    network1.add_edge(3, 2, randint(100, 1000))
    network1.add_edge(3, 7, randint(100, 1000))
    network1.add_edge(4, 3, randint(100, 1000))
    network1.add_edge(5, 2, randint(100, 1000))
    network1.add_edge(5, 6, randint(100, 1000))
    network1.add_edge(6, 7, randint(100, 1000))

    s, t = 0, 7
    max_flow = network1.max_flow(s, t)
    S, T = network1.min_cut(s)
    print("\nПример со случайными значениями")
    print(f"Максимальный поток: {max_flow}")
    print(f"Минимальный разрез:")
    print(f"S = {S}")
    print(f"T = {T}")
    network1.verify_min_cut(max_flow, S, T)
