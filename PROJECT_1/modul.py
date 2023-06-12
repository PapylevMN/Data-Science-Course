import numpy as np

def outliers_z_score(data, feature, left= 3, right= 3, log_scale=False):
    """Выявляет выбросы в данных данных методом z-отклонений (метод сигм):
        Правило трёх сигм: если распределение данных является нормальным, то 99,73 % лежат в интервале от (мю - 3 * сигма),
        где (мю) — математическое ожидание (для выборки это среднее значение), а (сигма) — стандартное отклонение. 
        Наблюдения, которые лежат за пределами этого интервала, будут считаться выбросами.

    Args:
        data: (DataFrame): Исходный DataFrame
        feature (str): Признак по которому выявляются выбросы
        left (int): Левая граница интервала в количестве сигма. Default to 3
        right (int): Правая граница интервала в количестве сигма. Default to 3
        log_scale (bool): Если True, признак логарифмируется

    Returns:
        DataFrame: данные выявленные как выбросы
        DataFrame: очищенные от выбросов данные
    """
    if log_scale:
        x = np.log(data[feature]+1)
    else:
        x = data[feature]
    mu = x.mean()
    sigma = x.std()
    lower_bound = mu - left * sigma
    upper_bound = mu + right * sigma
    outliers = data[(x < lower_bound) | (x > upper_bound)]
    cleaned = data[(x > lower_bound) & (x < upper_bound)]
    return outliers, cleaned

if __name__ == "__main__":
    print('Вспомогательный модуль')