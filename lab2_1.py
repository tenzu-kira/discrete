from itertools import permutations

# Исходное слово
word = "КОМБИНАТОРИКА"

# Уникальные буквы и их количество
letters_count = {
    'К': 2,
    'О': 2,
    'М': 1,
    'Б': 1,
    'И': 2,
    'Н': 1,
    'А': 2,
    'Т': 1,
    'Р': 1
}

# Длина слова, которое нужно составить
word_length = 6

# Генерация всех возможных комбинаций
unique_words = set()

# Используем permutations для генерации всех возможных упорядоченных комбинаций
for combo in permutations(word, word_length):
    # Проверяем, что комбинация не превышает доступное количество букв
    valid = True
    for letter in set(combo):
        if combo.count(letter) > letters_count.get(letter, 0):
            valid = False
            break
    if valid:
        unique_words.add(combo)

# Вывод результата
print(f"Количество различных слов: {len(unique_words)}")
