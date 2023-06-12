""" Игра угадай число:
    Компьютер сам загадывает и сам угадывает число
"""

import numpy as np

def find_number(number: int = 1) -> int:
    """Угадываем число

    Args:
        number (int, optional): Загаданное число. Defaults to 1.

    Returns:
        int: Число попыток
    """
    count = 0
    low = 0
    high = 101
    while True:
        count += 1
        predict_number = (low + high)//2       # Угадываем бинарным поиском
        if number > predict_number:
            low = predict_number
        elif number < predict_number:
            high = predict_number 
        else:
            break                              # выход из цикла если угадали
    return count


def score_game(find_number) -> int:
    """За какое количство попыток в среднем за 1000 подходов угадывает наш алгоритм

    Args:
        find_number([type]): функция угадывания

    Returns:
        int: среднее количество попыток
    """
    count_ls = []
    random_array = np.random.randint(1, 101, size=(1000))  # загадали список чисел
    for number in random_array:
        count_ls.append(find_number(number))

    score = int(np.mean(count_ls))
    print(f"Ваш алгоритм угадывает число в среднем за:{score} попыток")
    return score

if __name__ == "__main__":
    score_game(find_number)

