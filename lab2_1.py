from itertools import product

word = "КОМБИНАТОРИКА"
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

word_length = 6

unique_letters = list(letters_count.keys())

# Функция для проверки, что комбинация не превышает доступное количество букв
def is_valid(combo):
    for letter in set(combo):
        if combo.count(letter) > letters_count[letter]:
            return False
    return True

valid_combinations = set()

# Используем product для генерации всех возможных комбинаций
for combo in product(unique_letters, repeat=word_length):
    if is_valid(combo):
        valid_combinations.add(combo)

print(f"Количество различных слов: {len(valid_combinations)}")
