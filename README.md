# vkr
ВКР по курсу Python
Для защиты 20/02/2020

"""
Модуль defolt_picture.py предназначен для:
- подбора рейтингов в заданном массиве данных по платежам,
- прогнозирования картины дефолтов  в зависимости от рейтинга

Методы для подготовки данных и параметров:
- read_data - считывает данные из исходного файла с платежами в ДатаФрейм
- get_rating_target_values - задает целевые доли рейтингов по годам. Возвращается словарь.
- get_backet_target_values - задает целевые параметры бакетов по годам. Возвращает сложный словарь.

Основной метод:
- construct_default_picture - по заданному алгоритму строит картину платежей. Возвращет ДатаФрейм.

Методы оценки результата:
- get_rating_result_values
- get_backet_result_values
"""
