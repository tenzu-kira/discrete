import numpy as np
from itertools import product


def polynomial_to_vector(poly_str, degree):
    """Преобразует строку многочлена в вектор коэффициентов."""
    return [int(bit) for bit in poly_str.ljust(degree + 1, '0')[:degree + 1]][::-1]


def polynomial_mod(poly, mod_poly):
    """Делит poly на mod_poly и возвращает остаток."""
    poly = poly.copy()
    while len(poly) >= len(mod_poly):
        if poly[-1]:
            shift = len(poly) - len(mod_poly)
            for i in range(len(mod_poly)):
                poly[shift + i] ^= mod_poly[i]
        poly = poly[:-1]
    return poly


def build_generator_matrix(n, k, g_poly_str):
    """Строит порождающую матрицу G для циклического кода."""
    g_poly = polynomial_to_vector(g_poly_str, n - k)
    G = []
    for i in range(k):
        info_vec = [0] * k
        info_vec[i] = 1
        xr_poly = [0] * (n - k) + info_vec
        remainder = polynomial_mod(xr_poly, g_poly)
        row = info_vec + remainder[::-1]
        G.append(row)
    return np.array(G, dtype=int)


def build_parity_check_matrix(n, k, g_poly_str):
    """Строит проверочную матрицу H для циклического кода в систематической форме."""
    r = n - k
    g_poly = polynomial_to_vector(g_poly_str, r)

    # Инициализируем матрицу H нулями
    H = [[0] * n for _ in range(r)]

    # Заполняем левую часть (P^T)
    for i in range(k):
        # Вычисляем x^(k+i) mod g(x)
        info_vec = [0] * k
        info_vec[i] = 1
        xr_poly = [0] * (n - k) + info_vec
        remainder = polynomial_mod(xr_poly, g_poly)[::-1]

        # Записываем остаток в соответствующие строки H
        for row in range(r):
            H[row][i] = remainder[row] if row < len(remainder) else 0

    # Добавляем единичную матрицу справа (I_r)
    for i in range(r):
        H[i][k + i] = 1

    return np.array(H, dtype=int)

def compute_syndrome(received_vec, H):
    """Вычисляет синдром принятого вектора."""
    return tuple(np.dot(H, received_vec) % 2)


def build_syndrome_table(n, H):
    """Строит таблицу синдромов для одиночных ошибок."""
    syndrome_table = {}
    for error_pos in range(n):
        error_vec = np.zeros(n, dtype=int)
        error_vec[error_pos] = 1
        syndrome = compute_syndrome(error_vec, H)
        syndrome_table[syndrome] = error_pos
    return syndrome_table


def correct_errors(received_vec, H, syndrome_table):
    """Обнаруживает и исправляет ошибки в принятом векторе."""
    syndrome = compute_syndrome(received_vec, H)
    if sum(syndrome) == 0:
        return received_vec, "No errors detected"

    if syndrome in syndrome_table:
        error_pos = syndrome_table[syndrome]
        corrected_vec = received_vec.copy()
        corrected_vec[error_pos] ^= 1
        return corrected_vec, f"Corrected error at position {error_pos}"
    else:
        return received_vec, "Error detected but cannot be corrected"


# Параметры кода
n = 21  # Длина кодового слова
k = 15  # Длина информационного слова
g_poly_str = "1010111"  # g(x) = x^6 + x^4 + x^2 + x + 1

# Построение матриц
G = build_generator_matrix(n, k, g_poly_str)
H = build_parity_check_matrix(n, k, g_poly_str)
syndrome_table = build_syndrome_table(n, H)




# Генерация всех кодовых слов
def generate_all_codewords(G):
    k = G.shape[0]
    all_codewords = []
    for bits in product([0, 1], repeat=k):
        codeword = np.zeros(G.shape[1], dtype=int)
        for i in range(k):
            if bits[i]:
                codeword = (codeword + G[i]) % 2
        all_codewords.append(codeword)
    return np.array(all_codewords)


codewords = generate_all_codewords(G)
print("\nВсего кодовых слов:", len(codewords))


# Вычисление минимального расстояния
def compute_min_distance(codewords):
    min_dist = n
    for i in range(1, len(codewords)):
        weight = sum(codewords[i])
        if weight < min_dist:
            min_dist = weight
    return min_dist


# Пример использования
true_codeword = codewords[19000]
received_vec = true_codeword.copy()
received_vec[3] ^= 1  # Добавляем ошибку на 3-й позиции
# Исправление ошибки
corrected_vec, message = correct_errors(received_vec, H, syndrome_table)
print("Received:", received_vec)
print("Corrected:", corrected_vec)
print("Status:", message)
print("Match original:", np.array_equal(corrected_vec, true_codeword))


min_distance = compute_min_distance(codewords)
print("Минимальное кодовое расстояние:", min_distance)
print(G[1][1])
