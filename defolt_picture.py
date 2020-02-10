#!/usr/bin/env python
# coding: utf-8

"""
Модуль предназначен для:
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
import datetime
from dateutil.parser import parse
from numpy import nan
import pandas as pd
import os
from typing import Dict, NewType
from collections import namedtuple

RatingNames = namedtuple('RatingNames', [
    'high', 'medium', 'low', 'express', 'without'])
    
    
Borders = namedtuple('Borders', ['min', 'max'])
Encodings = namedtuple('Encodings', ['utf8', 'win1251'])
DFrame = NewType('DFrame', pd.core.frame.DataFrame)

RATING_NAMES = RatingNames(
    high="высокий",
    medium="средний",
    low="низкий",
    express="экспресс без ФА",
    without='без рейтинга')
    
LIST_OF_ENCODINGS = Encodings(utf8='utf-8', win1251='Windows-1251')

DEFAULT_FILENAME = os.path.join(os.path.dirname(__file__), "sample.csv")


def timer(fn):
    def wrapper(*args, **qwargs):
        """
        Декоратор для измерения времени исполнения функции
        """
        start = datetime.datetime.now()
        res = fn(*args, **qwargs)
        finish = datetime.datetime.now()
        print(f'Функция {fn.__name__} выполнилась за: {finish - start}')
        return res
    return wrapper


@timer
def main():
    """
    Описывает основной алгоритм:
    - получить целевые значения рейтингов
    - получить целевые значения бакетов
    - заполнить картину дефолтов
    - вывести результаты распределения рейтингов
    - вывести результаты распределения бакетов
    - сохранить результаты в csv
    """
    start = datetime.datetime.now()
    print(f'We start execution at: {start}')
    # csv = 'code/Подготовленный.csv'
    # df = read_data(filename=csv)
    df = read_data()
    rating_target = get_rating_target_values(rating_tobe_last=RATING_NAMES.high)
    backet_target = get_backet_target_values()
    df_with_backet = construct_default_picture(df, rating_target, backet_target)
    rating_result = get_rating_result_values(df_with_backet)
    print(f'Результаты подбора рейтингов: {rating_result}')
    rating_target = get_rating_target_per_years(df)
    print(f'Целевые значения рейтингов: {rating_target}')
    #TODO: backet_result = get_backet_result_values(df_with_backet)
    finish = datetime.datetime.now() - start
    print(f'We finished. Total time execution: {finish}')


def read_data(filename:str=None, encoding='utf-8')->DFrame:
    """
    read data from file
    return pandas Dataframe
    """

    global RATING_NAMES
    global DEFAULT_FILENAME
    global LIST_OF_ENCODINGS


    if filename is None:
        filename = DEFAULT_FILENAME
    _check_type(filename, str)
    if encoding is None:
        encoding = 'utf-8'
    _check_type(encoding, str)
    assert encoding in LIST_OF_ENCODINGS, f'Полученная кодировка \
        {encoding} отсутствует в списке возможных {LIST_OF_ENCODINGS}'
    csv = filename
    df = pd.read_csv(csv, encoding=encoding, sep=',', index_col=['Расчет', 'IDКлиента'])
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)
    df.fillna(value=nan, inplace=True)
    df.replace(0, nan, inplace=True)
    pd.options.display.float_format = '{:,.2f}'.format
    df['ratings'] = RATING_NAMES.without
    mask = (df['ratings'] == RATING_NAMES.without) & df['01.01.2018'].notna()
    return df


def get_rating_target_values(rating_tobe_last:str=None)->dict:
    """
    Возвращает список:
    0 - словарь с целевыми значениями долей рейтингов, ключи -названия рейтинга, значения экземпляр Borders
    1 - название ретийнга, который присваивается последним, по остаточному принципу
    """
    
    global RATING_NAMES #
    
    
    assert rating_tobe_last in RATING_NAMES, f'Полученный рейтинг \
                                              "{rating_tobe_last}" отсутствует в списке известных'
    rating_values = _get_min_max_rating_values()
    _check_type(rating_values, dict)
    if rating_tobe_last is None:
        rating_tobe_last = "экспресс без ФА"
    _check_type(rating_tobe_last, str)
    last_rating_to_define = rating_tobe_last
    rating_values[last_rating_to_define] = rating_values.pop(last_rating_to_define)

    return rating_values, last_rating_to_define


def get_backet_target_values()->dict:
    """
    Возвращает заданные возможные статусы платежа и их вероятности:
    "выход на просрочку":{
        "Без просрочки": ("1-30"  , rating_dependencies),
        },
    "увеличение просрочки": {
            "1-30"         : ("31-60" , 0.25),
            "31-60"        : ("дефолт", 0.3),
        },
    "погашение просрочки": {
            "1-30"         : ("Без просрочки", 0.25),
            "31-60"        : ("1-30"         , 0.4),
        },

    """
    rating_probability_tobe_overdue = _get_rating_dependencies()
    _check_type(rating_probability_tobe_overdue, dict)
    payment_movements_probability = _get_payment_statuses(rating_probability_tobe_overdue)
    _check_type(payment_movements_probability, dict)
    
    return payment_movements_probability


def construct_default_picture(df_to_fill:DFrame, rating_target: dict, backet_target: dict)->DFrame:
    """
    основной этап: для каждого месяца заполняет рейтинг, строит картину платежей
    """
    global RATING_NAMES

    _check_type(df_to_fill, pd.core.frame.DataFrame)
    _check_type(rating_target, tuple)
    _check_type(backet_target, dict)

    df = df_to_fill.copy()
    
    rating_values = rating_target[0]
    last_rating_to_define = rating_target[1]
    probability_tobe_overdue = backet_target['выход на просрочку'][RATING_NAMES.without][1]
    probability_tobe_overdue[last_rating_to_define] = probability_tobe_overdue.pop(last_rating_to_define)

    clients_with_rating = {}
    backets = []
    
    payment_statuses = backet_target

    for idx, month in enumerate(df.columns, 1):
        try: 
            parse(month)
        except Exception:
            continue
        backets, cur_backet_month, prev_backet_month = _get_cur_and_prev(backets, month)
        _fill_cur_backet_from_prev(df, month, cur_backet_month, prev_backet_month)

        for status, categories in payment_statuses.items():
            if status=="выход на просрочку":
                sum_month = round(df[month].sum(), 2)
                ratings_proceed = 0
                for rating, probability in probability_tobe_overdue.items():
                    # присваиваем рейтинги для всех, кроме "последнего"
                    if rating!=last_rating_to_define:
                        min_val = rating_values[rating].min
                        rating_share = _define_rating_share(
                            df=df, 
                            month=month, 
                            rating=rating, 
                            min_value=min_val,
                            clients_with_rating=clients_with_rating, 
                            sum_month=sum_month)
                        ratings_proceed += 1
                        # начинаем набор следующего бакета
                        rating_sum = round(df[df['ratings']==rating][month].sum(), 2)
                        target_share = probability
                        overdue_rating_sum = _fill_payment_overdue(
                            df=df, 
                            month=month, 
                            cur_backet_month=cur_backet_month, 
                            rating=rating, 
                            rating_sum=rating_sum, # может быть, надо заменить на rating_sum, так было
                            target_share=target_share)
                        share = overdue_rating_sum / rating_sum
                    # заполняем последний
                    else: 
                        no_rating_df = df[(df[month].notna()) & (df['ratings']==RATING_NAMES.without)]
                        df.loc[no_rating_df.index.values, 'ratings'] = last_rating_to_define
                    print(f' month: {month}, rating: {rating}')
            elif idx > 1:
                for category, chance in categories.items():
                    category_to_set = chance[0]        
                    target_share = chance[1]
                    cat_in_prev_backet = df[(df[month].notna()) & \
                                            (df[prev_backet_month]==category)]
                    sum_cat_in_prev_backet = round(cat_in_prev_backet[month].sum(), 2)
                    share_category = 0
                    if sum_cat_in_prev_backet > 0:
                        sum_cat_in_cur_backet = _fill_payment_progress(
                            df=df,
                            month=month,
                            cur_backet_month=cur_backet_month,
                            cat_in_prev_backet=cat_in_prev_backet, 
                            sum_cat_in_prev_backet=sum_cat_in_prev_backet, 
                            category_to_set=category_to_set, 
                            target_share=target_share)
                        share_category = sum_cat_in_cur_backet / sum_cat_in_prev_backet
                    # print(f'from {prev_backet_month} and {category}-{sum_cat_in_prev_backet:,.0f} to {cur_backet_month}-{category_to_set}-{sum_cat_in_cur_backet:,.0f}, share: {share_category:.0%}')
        k = _get_adjusting_coefficient(month)                                          
        rating_values = _correct_min_max_values(rating_values, k)
    return df


def get_rating_result_values(df:DFrame)->DFrame:
    """
    Возвращеет статистику по долям рейтингов заданном датафрейме
    """
    shares = df.groupby('ratings').sum() / df.groupby('ratings').sum().sum()
    shares = pd.DataFrame(shares).T
    shares = shares.set_index(pd.to_datetime(shares.index, dayfirst=True))
    years = set([month.year for month in shares.index])
    return pd.DataFrame({year_: shares.loc[str(year_), :].mean() for year_ in years})


def get_rating_target_per_years(df:DFrame, rating_val:dict)->DFrame:
    """
    Возвращает ДатаФрейм, содержащий цлеевые значения долей рейтингов по годам
    """
    res = {}
    rt_val = rating_val.copy()
    for month in df.columns:
        try: 
            parse(month)
        except Exception:
            continue
            
        res[month] = {rating: value.min for rating,value in rt_val.items()}
        k = _get_adjusting_coefficient(month)
        rt_val = _correct_min_max_values(rt_val, k)
    years = set([parse(month).year for month in res])
    res = pd.DataFrame(res).T
    res.index=pd.to_datetime(res.index, dayfirst=True)
    res.loc[:, 'экспресс без ФА'] = 1 - (
        res.loc[:, 'низкий'] +
        res.loc[:, 'средний'] +
        res.loc[:, 'высокий']
    )
    res = {year_: res.loc[str(year_), :].mean() for year_ in years}
    return pd.DataFrame(res)


def _check_collection_for_ratings(collection)->None:
    global RATING_NAMES
    for name in RATING_NAMES:
        if name != RATING_NAMES.without:
            assert name in collection, f'В справочнике {collection} отсутствует рейтинг {name}'


def _check_type(var, expected_type):
    """
    проверяет соответствие типа полученной переменной полученному типу.
    возбуждает AssertException при ошибке
    """
    assert isinstance(var, expected_type), f"Некорректный тип: {type(var)}"


def _get_min_max_rating_values(data=None) -> dict:
    """
        returns dictionary wich contains inner dictionary with 2 keys: min and max:
        Usage:
            ratings_values = get_min_max_rating_values()
            grade_high_min_value = ratings_values['высокий'].min
            grade_high_max_value = ratings_values['высокий'].max
        }
    """
    if data is None:
        min_max_values = {
            'экспресс без ФА': Borders(min=0.02, max=0.03),
            'низкий':          Borders(min=0.20, max=0.22),
            'средний':         Borders(min=0.24, max=0.26),
            'высокий':         Borders(min=0.53, max=0.56),
        }

    _check_collection_for_ratings(min_max_values)

    return min_max_values


def _get_adjusting_coefficient(month)->dict:
    """
    Возращаем коэффициент коррекции в зависимости от полученного месяца
    """
    
    global RATING_NAMES

    assert isinstance(month, str), f'Некорректный тип: {type(month)}'

    trgt_years_rating = {
        2018: {
            RATING_NAMES.high:    0.52,
            RATING_NAMES.medium:  0.26,
            RATING_NAMES.low:     0.20,
            RATING_NAMES.express: 0.02,
        },
        2019: {
            RATING_NAMES.high:    0.50,
            RATING_NAMES.medium:  0.27,
            RATING_NAMES.low:     0.21,
            RATING_NAMES.express: 0.02,
        },
        2020: {
            RATING_NAMES.high:    0.46,
            RATING_NAMES.medium:  0.30,
            RATING_NAMES.low:     0.22,
            RATING_NAMES.express: 0.02,
        },
        2021: {
            RATING_NAMES.high:    0.40,
            RATING_NAMES.medium:  0.34,
            RATING_NAMES.low:     0.24,
            RATING_NAMES.express: 0.02,
        },
        2022: {
            RATING_NAMES.high:    0.34,
            RATING_NAMES.medium:  0.35,
            RATING_NAMES.low:     0.25,
            RATING_NAMES.express: 0.06,
        }, 
    }
    try:
        _date = parse(month)
    except Exception:
        f'Ошибка при парсинге полученного месяца: {month}'
    return trgt_years_rating[_date.year]


def _correct_min_max_values(rating_values, trgt_val)->dict:
    """
    Корректирует полученные значения рейтингов на полученный коэффицент.
    Возвращает скорректированные значения рейтингов.
    """

    global RATING_NAMES


    assert isinstance(rating_values, dict), f'Некорректный тип {type(rating_values)}'
    _check_collection_for_ratings(rating_values)
    assert isinstance(trgt_val, dict), f'Некорректный тип {type(trgt_val)}'

    rating_values['высокий'] = Borders(
        min=trgt_val[RATING_NAMES.high], 
        max=trgt_val[RATING_NAMES.high] + 0.01,
        )
    
    rating_values['средний'] = Borders(
        min=trgt_val[RATING_NAMES.medium], 
        max=trgt_val[RATING_NAMES.medium] + 0.01,
        )

    rating_values['низкий'] = Borders(
        min=trgt_val[RATING_NAMES.low], 
        max=trgt_val[RATING_NAMES.low] + 0.01,
        )

    return rating_values


def _get_rating_dependencies()->dict:
    """
    Возвращает словарь с ключами - названиями рейтингов и значениями - целевыми значениями рейтинга
    """
    k = 0.35
    return {
        "высокий": round(k * 0.04, 5),
        "средний": round(k * 0.08, 5),
        "низкий": round(k * 0.12, 5),
        "экспресс без ФА": round(k * 0.10, 5)
    }


def _get_payment_statuses(rating_dependencies:dict)->dict:
    """
    Получает целевые параметры перехода просрочки из бакета в бакет
    """
    global RATING_NAMES
    if rating_dependencies is None:
        rating_dependencies = _get_rating_dependencies()
    assert isinstance(rating_dependencies, dict), f'Некорректный тип: {type(rating_dependencies)}'
    
    _check_collection_for_ratings(rating_dependencies)
    
    return {
        "выход на просрочку": {
            RATING_NAMES.without: ("1-30", rating_dependencies),
        },
        "увеличение просрочки": {
            "1-30": ("31-60", 0.25),
            "31-60": ("дефолт", 0.3),
        },
        "погашение просрочки": {
            "1-30": ("Без просрочки", 0.25),
            "31-60": ("1-30", 0.4),
        },
    }


def _define_rating_share(
    df: DFrame, 
    month:str, 
    rating:str, 
    min_value:float, 
    clients_with_rating:dict, 
    sum_month:float)->float:
    global RATING_NAMES
    """
    Заполняет рейтинг в заданном месяце.
    Возвращет итоговое значение доли рейтинга.
    """
    assert isinstance(df, pd.core.frame.DataFrame), f"Некорректный тип: {type(df)}"
    assert isinstance(month, str), f"Некорректный тип: {type(month)}"
    assert isinstance(rating, str), f"Некорректный тип: {type(rating)}"
    assert isinstance(min_value, float), f"Некорректный тип: {type(min_value)}"

    _min = min_value
    mask = (df['ratings'] == rating)
    sum_rating = round(df[mask][month].sum(), 2)
    if sum_rating / sum_month < _min:
        without_rating = (df[month].notna()) & (df['ratings'] == RATING_NAMES.without)
        sum_rating, clients_with_rating = _fill_rating(
            df=df,
            month=month,
            empty_criteria=without_rating, 
            rating=rating, 
            sum_rating=sum_rating, 
            sum_month=sum_month, 
            min_value=_min, 
            proceed_clients=clients_with_rating)
    return sum_rating / sum_month


def _fill_rating(df, month, empty_criteria, rating, sum_rating, sum_month, min_value, proceed_clients):
    """
    Заполняет рейтинги в заданных параметрах
    """
    empty_data = df[empty_criteria]
    # это быстрее, чем empty_data.shape[0] на 30 - 40%%
    empty_rows_count = len(empty_data)
    cur_share = sum_rating / sum_month
    step = 1
    while cur_share < min_value and step <= empty_rows_count:
        # idx = rnd.randint(0, empty_rows_count - 1)
        # id_client = empty_data.index.values[idx][1]
        id_client = empty_data.sample(1).index[0][1]
        if id_client not in proceed_clients:
            proceed_clients[id_client] = month
        elif proceed_clients[id_client] != month:
            raise Exception(
                'Клиент повторно обрабатывается из предыдущего месяца!')
        else:
            continue
        df.loc[(slice(None), id_client), 'ratings'] = rating
        sum_rating += round(df.loc[(slice(None), id_client), month].sum(), 2)
        cur_share = sum_rating / sum_month
        step += 1
    return sum_rating, proceed_clients


def _fill_payment_overdue(df, month, cur_backet_month, rating, rating_sum, target_share):
    """
    Присваивает платежу статус первой просрочки
    """
    global RATING_NAMES
    cur_category = RATING_NAMES.without
    target_category = "1-30"
    proceeded_clients = {}
    no_debt_df = df[(df[month].notna()) & (df[cur_backet_month]
                                           == cur_category) & (df['ratings'] == rating)]
    no_debt_length = len(no_debt_df)
    try:
        sum_target_category = df.groupby([cur_backet_month, 'ratings'])[
            month].sum().loc[target_category, rating]
    except KeyError:
        sum_target_category = 0
    cur_share = sum_target_category / rating_sum
    step = 1
    while cur_share < target_share and step <= no_debt_length:
        # idx = rnd.randint(0, no_debt_length - 1)
        # id_client = no_debt_df.index.values[idx][1]
        id_client = no_debt_df.sample(1).index[0][1]
        if id_client not in proceeded_clients:
            proceeded_clients[id_client] = target_category
        else:
            continue
        df.loc[(slice(None), id_client),
               cur_backet_month] = target_category
        sum_target_category += round(
            df.loc[(slice(None), id_client), month].sum(), 2)
        cur_share = sum_target_category / rating_sum
        step += 1
    return sum_target_category


def _fill_payment_progress(
    df,
    month, 
    cur_backet_month,
    cat_in_prev_backet, 
    sum_cat_in_prev_backet, 
    category_to_set, 
    target_share):
    """
    присваивает просроченному платежу очередной бакет
    """
    max_count = len(cat_in_prev_backet)
    try:
        sum_cat_in_cur_backet = cat_in_prev_backet.groupby(cur_backet_month)[month].sum().loc[category_to_set]
    except KeyError:
        sum_cat_in_cur_backet = 0
    share = sum_cat_in_cur_backet / sum_cat_in_prev_backet
    step = 1
    proceeded_clients = {}
    while share < target_share and step <= max_count:
        # idx = rnd.randint(0, max_count - 1)
        # id_client = cat_in_prev_backet.index.values[idx][1]
        id_client = cat_in_prev_backet.sample(1).index[0][1]
        if id_client not in proceeded_clients:
            proceeded_clients[id_client] = category_to_set
        else:
            continue
        df.loc[(slice(None), id_client),
               cur_backet_month] = category_to_set
        sum_cat_in_cur_backet += round(
            df.loc[(slice(None), id_client), month].sum(), 2)
        share = sum_cat_in_cur_backet / sum_cat_in_prev_backet
        step += 1
    return sum_cat_in_cur_backet


def _get_cur_and_prev(backets:list, month:str)->list:
    """
    Добавляем в полученный список название теекущего месяца с бакетами,
    возвращаем:
    (обновленный список,
     название текущего столбца,
     название предыдущего столбца)
    """
    current = 'backet_' + month
    backets.append(current)
    previous = backets[-2] if len(backets) > 1 else current
    return backets, current, previous


def _fill_cur_backet_from_prev(
    df: DFrame,
    month: str,
    cur_backet_month: str,
    prev_backet_month: str)->str:
    """

    """
    global RATING_NAMES
    df[cur_backet_month] = ""
    df[cur_backet_month] = df[prev_backet_month]
    no_backet_df = df[(df[month].notna()) & (df[cur_backet_month] == "")]
    df.loc[no_backet_df.index.values, cur_backet_month] = RATING_NAMES.without
    res = f'Текущий {cur_backet_month} предварительно заполнен.'
    return res


def get_backet_resulting_shares(df):
    return df.loc[:2000, [
        'backet_01.01.2019',
        'backet_01.02.2019',
        'backet_01.03.2019',
        'backet_01.04.2019',
        'backet_01.05.2019',
        'backet_01.06.2019',
        'backet_01.07.2019',
        'backet_01.08.2019',
        'backet_01.09.2019',
        'backet_01.10.2019',
        'backet_01.11.2019',
        'backet_01.12.2019',
    ]]


if __name__ == '__main__':
    main()
    # ## Тайминги исполнения:
    #
    # ### Сперва присваивается "Экспресс без ФА"
    # + 0:01:02.029426
    # + 0:01:02.458878
    # + 0:01:03.278037
    #
    # #### Добавлены дебажные словари:
    # + 0:01:06.061935
    # + 0:01:10.485903
    #
    # ### Сперва присваивается "Высокий"