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


def print_code_distance_table(codewords, max_rows=20, exclude_zero=True):
    """
    Выводит фрагмент таблицы кодовых расстояний между случайными кодовыми словами.

    Параметры:
        codewords (np.array): Массив всех кодовых слов
        max_rows (int): Максимальное количество строк для вывода
        exclude_zero (bool): Исключать ли нулевое кодовое слово из сравнений
    """
    # Убираем нулевое слово при необходимости
    filtered_codewords = codewords[1:] if exclude_zero else codewords
    num_codewords = len(filtered_codewords)

    print("Фрагмент таблицы кодовых расстояний (случайные пары):")
    print(f"{'Кодовое слово 1':<20} {'Кодовое слово 2':<20} {'Расстояние':<10}")
    print("-" * 55)

    # Выбираем случайные пары для демонстрации
    rng = np.random.default_rng()
    indices = rng.choice(num_codewords, size=min(max_rows * 2, num_codewords), replace=False)

    count = 0
    for i in range(0, len(indices), 2):
        if i + 1 >= len(indices):
            break

        cw1 = filtered_codewords[indices[i]]
        cw2 = filtered_codewords[indices[i + 1]]
        distance = np.sum(cw1 != cw2)

        # Красивое представление векторов
        str_cw1 = ''.join(map(str, cw1))
        str_cw2 = ''.join(map(str, cw2))

        print(f"{str_cw1:<20} {str_cw2:<20} {distance:<10}")
        count += 1
        if count >= max_rows:
            break


# Пример использования
# Вычисляем минимальное расстояние кода
min_distance = compute_min_distance(codewords)
print(f"Минимальное расстояние кода (d): {min_distance}")

# Вычисляем кратность исправляемых ошибок
t = (min_distance - 1) // 2
print(f"Кратность гарантированно исправляемых ошибок (t): {t}")

# Вычисляем кратность обнаруживаемых ошибок
s = min_distance - 1
print(f"Кратность гарантированно обнаруживаемых ошибок (s): {s}")

# Создаем кодовое слово (первый информационный бит = 1)
info_bits = np.zeros(k, dtype=int)
info_bits[0] = 1
codeword = (info_bits @ G) % 2

# Создаем ошибку в позиции 5
error = np.zeros(n, dtype=int)
error[5] = 1
received = (codeword + error) % 2
syndrome = compute_syndrome(error, H)
syndrome_out = [int(i) for i in syndrome]
# Исправляем ошибку
corrected, message = correct_errors(received, H, syndrome_table)
print("\nПример 1: Исправление одиночной ошибки")
print("Переданное кодовое слово:", codeword)
print("Вектор ошибки:", error)
print("Принятый вектор:", received)
print("Результат коррекции:", message)
print("Исправленный вектор:", corrected)
print("Синдром:", syndrome_out)

# Создаем двойную ошибку (позиции 3 и 7)
error = np.zeros(n, dtype=int)
error[3] = 1
error[7] = 1
received = (codeword + error) % 2
syndrome = compute_syndrome(error, H)
syndrome_out = [int(i) for i in syndrome]
# Пытаемся исправить
corrected, message = correct_errors(received, H, syndrome_table)
print("\nПример 2: Обнаружение двойной ошибки")
print("Переданное кодовое слово:", codeword)
print("Вектор ошибки:", error)
print("Принятый вектор:", received)
print("Результат коррекции:", message)
print("Синдром:", syndrome_out)

# Создаем вектор двойной ошибки
error = np.zeros(n, dtype=int)
error[0] = 1  # Ошибка в позиции 0
error[10] = 1  # Ошибка в позиции 10

# Вычисляем синдром
syndrome = compute_syndrome(error, H)
syndrome_out = [int(i) for i in syndrome]
print("\nВектор ошибки, который обнаруживается, но не исправляется:")
print("Вектор ошибки:", error)
print("Синдром:", syndrome_out)
print("Синдром есть в таблице?", syndrome in syndrome_table)

# Проверяем, что эта ошибка действительно не исправляется
received = (codeword + error) % 2
corrected, message = correct_errors(received, H, syndrome_table)
print("Результат попытки коррекции:", message)
print()
print_code_distance_table(codewords)
print()
print("Порождающая матрица:")
print(G)
