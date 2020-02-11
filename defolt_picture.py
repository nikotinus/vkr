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
import logging
import os
import pandas as pd
import random as rnd
from collections import namedtuple
from dateutil.parser import parse, ParserError
from numpy import nan
from typing import Dict, NewType

from my_logging import get_func_name, get_func_params_and_values, get_func_params


# Блок общего описания
name = 'default_picture'
version = 2.4
updates = f"""
2.4:
    - small bugfix and making code nice
2.3:
    - bug fixed: backet name "без рейтинга" исправлен на "Без просрочки"
2.2:
    - asserts moved into DEBUG section
"""
instr = f'''
    Версия: {name}-{version}
    {__doc__}
'''

# Блок логгирования
# log_format = u'[LINE:%(lineno)d]# %(levelname)-8s %(message)s'
log_format = u'%(levelname)-8s %(message)s'
log_level = logging.INFO
path = 'Logs'
name = f'logs_{name}_{version}.log'
log_file_name = os.path.join(path, name)
logging.basicConfig(format=log_format, level=log_level, filename=log_file_name)


RatingNames = namedtuple('RatingNames', 
    ['high', 'medium', 'low', 'express', 'without'])
Borders = namedtuple('Borders', ['min', 'max'])
Encodings = namedtuple('Encodings', ['utf8', 'win1251'])
Backetnames = namedtuple('Backetnames', ['intime', 'first', 'second', 'defolt'])
DFrame = NewType('DFrame', pd.core.frame.DataFrame)

RATING_NAMES = RatingNames(
    high="высокий",
    medium="средний",
    low="низкий",
    express="экспресс без ФА",
    without='без рейтинга')

TRGT_YEARS_RATING = {
        2018: {
            RATING_NAMES.high:    Borders(min=0.52, max=0.56),
            RATING_NAMES.medium:  Borders(min=0.26, max=0.28),
            RATING_NAMES.low:     Borders(min=0.20, max=0.22),
            RATING_NAMES.express: Borders(min=0.02, max=0.03),
        },
        2019: {
            RATING_NAMES.high:    Borders(min=0.50, max=0.52),
            RATING_NAMES.medium:  Borders(min=0.27, max=0.29),
            RATING_NAMES.low:     Borders(min=0.21, max=0.23),
            RATING_NAMES.express: Borders(min=0.02, max=0.03),
        },
        2020: {
            RATING_NAMES.high:    Borders(min=0.46, max=0.49),
            RATING_NAMES.medium:  Borders(min=0.30, max=0.32),
            RATING_NAMES.low:     Borders(min=0.22, max=0.24),
            RATING_NAMES.express: Borders(min=0.02, max=0.03),
        },
        2021: {
            RATING_NAMES.high:    Borders(min=0.40, max=0.44),
            RATING_NAMES.medium:  Borders(min=0.34, max=0.36),
            RATING_NAMES.low:     Borders(min=0.24, max=0.26),
            RATING_NAMES.express: Borders(min=0.02, max=0.03),
        },
        2022: {
            RATING_NAMES.high:    Borders(min=0.34, max=0.36),
            RATING_NAMES.medium:  Borders(min=0.35, max=0.38),
            RATING_NAMES.low:     Borders(min=0.25, max=0.27),
            RATING_NAMES.express: Borders(min=0.06, max=0.08),
        }, 
    }
BACKET_NAMES = Backetnames(
    intime = "Без просрочки", 
    first  = "1-30", 
    second = "31-60", 
    defolt = "defolt"
    )

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
        msg = f"{finish}, время исполнения: {finish - start}, функция {fn.__name__}"
        tmp = dict(**qwargs)
        if 'month' in tmp:
            msg = f"{finish}, время исполнения: {finish - start}, {tmp['month']}. Время исполнения функции: {fn.__name__}"
        logging.info(msg)
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
    msg = instr + updates
    logging.info(msg)
    # csv = 'code/Подготовленный.csv'
    # df = read_data(filename=csv)
    df = read_data()
    rating_target = get_rating_target_values(rating_tobe_last=RATING_NAMES.high)
    backet_target = get_backet_target_values()
    df_with_backet = construct_default_picture(df, rating_target, backet_target)
    rating_result = get_rating_result_values(df_with_backet)
    msg = f'Результаты подбора рейтингов:\n{rating_result}'
    logging.info(msg)
    rating_target = get_rating_target_per_years()
    msg = f'Целевые значения рейтингов:\n{rating_target}'
    logging.info(msg)
    #TODO: backet_result = get_backet_result_values(df_with_backet)
    finish = datetime.datetime.now() - start
    df_with_backet.to_csv('result.csv')
    print(f'We finished. Total time execution: {finish}')


@timer
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
    if log_level == logging.DEBUG: 
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
    # mask = (df['ratings'] == RATING_NAMES.without) & df['01.01.2018'].notna()
    return df


@timer
def get_rating_target_values(rating_tobe_last:str=None)->dict:
    """
    Возвращает список:
    0 - словарь с целевыми значениями долей рейтингов, ключи -названия рейтинга, значения экземпляр Borders
    1 - название ретийнга, который присваивается последним, по остаточному принципу
    """
    
    global RATING_NAMES #
    
    if log_level == logging.DEBUG: 
        assert rating_tobe_last in RATING_NAMES, f'Полученный рейтинг \
                                              "{rating_tobe_last}" отсутствует в списке известных'
    rating_values = _get_min_max_rating_values()
    if log_level == logging.DEBUG: 
        _check_type(rating_values, dict)
    if rating_tobe_last is None:
        rating_tobe_last = "экспресс без ФА"
    if log_level == logging.DEBUG: 
        _check_type(rating_tobe_last, str)
    last_rating_to_define = rating_tobe_last
    rating_values[last_rating_to_define] = rating_values.pop(last_rating_to_define)

    return rating_values, last_rating_to_define


@timer
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
    if log_level == logging.DEBUG: 
        _check_type(rating_probability_tobe_overdue, dict)
    
    payment_movements_probability = _get_payment_statuses(rating_probability_tobe_overdue)
    if log_level == logging.DEBUG: 
        _check_type(payment_movements_probability, dict)
    
    return payment_movements_probability


@timer
def construct_default_picture(df_to_fill:DFrame, rating_target: dict, backet_target: dict)->DFrame:
    """
    основной этап: для каждого месяца заполняет рейтинг, строит картину платежей
    """
    global RATING_NAMES

    if log_level == logging.DEBUG: 
        _check_type(df_to_fill, pd.core.frame.DataFrame)
        _check_type(rating_target, tuple)
        _check_type(backet_target, dict)

    df = df_to_fill.copy()
    
    rating_values = rating_target[0]

    last_rating_to_define = rating_target[1]
    probability_tobe_overdue = backet_target['выход на просрочку'][BACKET_NAMES.intime][1]
    probability_tobe_overdue[last_rating_to_define] = probability_tobe_overdue.pop(last_rating_to_define)

    backets = []
    
    payment_statuses = backet_target

    df_columns = df.columns
    for idx, month in enumerate(df_columns, 1):
        try: 
            _date = parse(month)
        except Exception:
            continue
        
        msg = f'Проверка максимального и минимального значений'

        backets, cur_backet_month, prev_backet_month = _get_cur_and_prev(backets, month)
        _fill_cur_backet_from_prev(df, month, cur_backet_month, prev_backet_month)

        for status, categories in payment_statuses.items():
            if status=="выход на просрочку":
                sum_month = round(df[month].sum(), 2)
                ratings_proceed = 0
                rating_values = _refresh_min_max_values(_date)
                for rating, probability in probability_tobe_overdue.items():
                    min_val = rating_values[rating].min
                    
                    msg = f'Минимальное значение за {month}: {min_val}'
                    logging.debug(msg)
                    
                    if rating!=RATING_NAMES.without:
                        if rating!=last_rating_to_define:
                            rating_share = _define_rating_share(
                                df=df, 
                                month=month, 
                                rating=rating, 
                                min_value=min_val,
                                sum_month=sum_month)
                            ratings_proceed += 1

                            # начинаем набор следующего бакета
                            rating_sum = round(df[['ratings', month]].groupby('ratings').sum().loc[rating][0], 2)
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
                if log_level == logging.DEBUG: 
                    rating_sums = df.groupby('ratings')[month].sum()
                    total_sum = rating_sums.sum()
                    assert round(sum_month,2)==round(total_sum,2), "sum_month {sum_month} и total_sum {total_sum} не равны"
                    rating_shares = rating_sums / total_sum
                    msg = f'{month}.:\n{rating_shares}'
                    logging.debug(msg)
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
    return df


@timer
def get_rating_result_values(df):
    """
    Возвращеет статистику по долям рейтингов в заданном датафрейме
    """
    global RATING_NAMES
    res = {}
    
    for month in df.columns:
        try: 
            _date = parse(month)
        except Exception:
            continue
        
        ratings_sums = df.groupby('ratings')[month].sum()
        total_sum = ratings_sums.sum()
        assert total_sum > 0, f'Сумма по месяцу {month} некорректна: {total_sum}'
        res[month] = {
            RATING_NAMES.high:    ratings_sums.loc[RATING_NAMES.high]    / total_sum,
            RATING_NAMES.medium:  ratings_sums.loc[RATING_NAMES.medium]  / total_sum ,
            RATING_NAMES.low:     ratings_sums.loc[RATING_NAMES.low]     / total_sum,
            RATING_NAMES.express: ratings_sums.loc[RATING_NAMES.express] / total_sum,
        }
    res = pd.DataFrame(res).T
    res.index=pd.to_datetime(res.index, dayfirst=True)
    
    return res.groupby(res.index.year).mean().T


@timer
def get_rating_target_per_years():
    global TRGT_YEARS_RATING
    indx = (RATING_NAMES.high, RATING_NAMES.medium, RATING_NAMES.low, RATING_NAMES.express)
    res = {year: {rating: value.min for rating, value in ratings.items()} for year,ratings in TRGT_YEARS_RATING.items()}
    res = pd.DataFrame(res, index=indx)
    return res


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

    global TRGT_YEARS_RATING


    if data is None:
        min_max_values = TRGT_YEARS_RATING[2018]
    
    if log_level == logging.DEBUG: 
        _check_collection_for_ratings(min_max_values)

    return min_max_values


def _refresh_min_max_values(month:datetime.datetime, trgt_years:dict=None)->dict:
    """
    Возращает целевые значения рейтинга в зависимости от месяца.
    Usage:
            ratings_values = _refresh_min_max_values(month)
            grade_high_min_value = ratings_values['высокий'].min
            grade_high_max_value = ratings_values['высокий'].max
    """
    
    global RATING_NAMES
    global TRGT_YEARS_RATING


    if log_level == logging.DEBUG: 
        assert isinstance(month, datetime.datetime), f'Некорректный тип: {type(month)}'
    if trgt_years is None:
        trgt_years = TRGT_YEARS_RATING
    if log_level == logging.DEBUG: 
        assert isinstance(trgt_years, dict), f'Некорректный тип: {type(trgt_years)}'
        assert month.year in trgt_years, f'Год полученной даты ({month}) отсутвует в целевых значениях {trgt_years}'

    min_max_values = trgt_years[month.year]
    
    if log_level == logging.DEBUG: 
        _check_collection_for_ratings(min_max_values)

    return min_max_values


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
    global BACKET_NAMES
    

    if rating_dependencies is None:
        rating_dependencies = _get_rating_dependencies()
    
    if log_level == logging.DEBUG: 
        assert isinstance(rating_dependencies, dict), f'Некорректный тип: {type(rating_dependencies)}'
        _check_collection_for_ratings(rating_dependencies)
    
    return {
        "выход на просрочку": {
            BACKET_NAMES.intime: (BACKET_NAMES.first, rating_dependencies),
        },
        "увеличение просрочки": {
            BACKET_NAMES.first:  (BACKET_NAMES.second, 0.25),
            BACKET_NAMES.second: (BACKET_NAMES.defolt, 0.3),
        },
        "погашение просрочки": {
            BACKET_NAMES.first:  (BACKET_NAMES.intime, 0.25),
            BACKET_NAMES.second: (BACKET_NAMES.first, 0.4),
        },
    }


def _define_rating_share(
    df: DFrame, 
    month:str, 
    rating:str, 
    min_value:float, 
    sum_month:float)->float:
    global RATING_NAMES
    """
    Заполняет рейтинг в заданном месяце.
    Возвращет итоговое значение доли рейтинга.
    """
    # _check_type(df, pd.core.frame.DataFrame)
    # _check_type(month, str)
    # _check_type(rating, str)
    # _check_type(min_value, float)

    mask = (df['ratings'] == rating)
    try:
        rating_sum = round(df[['ratings', month]].groupby('ratings').sum().loc[rating][0], 2)
    except KeyError:
        rating_sum = 0
    share = rating_sum / sum_month

    if share < min_value:
        rating_sum = _fill_rating(
            df=df,
            month=month, 
            rating=rating, 
            sum_rating=rating_sum, 
            sum_month=sum_month, 
            min_value=min_value)
        share = rating_sum / sum_month
    
    logging.debug(f'{month}. {rating}. {share:.2f}. {min_value:.2f}')
    return share


@timer
def _fill_rating(df, month, rating, sum_rating, sum_month, min_value):
    """
    Заполняет рейтинги в заданных параметрах
    """
    idx = pd.IndexSlice
    without_rating = (df[month].notna()) & (df['ratings'] == RATING_NAMES.without)

    unique_clients = pd.Series(df[without_rating].index.get_level_values('IDКлиента').unique()).to_list()
    if not unique_clients:
        msg = f'{month}. {rating}. Отсутствуют уникальные клиенты: {len(unique_clients)}'
        logging.info(msg)
    
    cur_share = sum_rating / sum_month

    while cur_share < min_value and unique_clients:
        logging.debug(f'{get_func_name()}.{month}. proceeding while: {cur_share} < {min_value}')
        id_client = rnd.choice(unique_clients)
        
        client_criteria = without_rating.values & (df.index.get_level_values(1)==id_client)
        df.loc[client_criteria, 'ratings'] = rating

        sum_rating += round(df[client_criteria].loc[(slice(None), id_client), month].sum(), 2)
        cur_share = sum_rating / sum_month
        
        msg = f'{get_func_name()}.{month}. рейтинг: {rating} в месяце: {month} имеет {cur_share:.2f} долю'
        logging.debug(msg)
        unique_clients.remove(id_client)

    return sum_rating


def _fill_payment_overdue(df, month, cur_backet_month, rating, rating_sum, target_share):
    """
    Присваивает платежу статус первой просрочки
    """
    global BACKET_NAMES
    cur_category = BACKET_NAMES.intime
    target_category = BACKET_NAMES.first
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
    global BACKET_NAMES


    df[cur_backet_month] = ""
    df[cur_backet_month] = df[prev_backet_month]
    no_backet_df = df[(df[month].notna()) & (df[cur_backet_month] == "")]
    df.loc[no_backet_df.index.values, cur_backet_month] = BACKET_NAMES.intime
    res = f'Текущий {cur_backet_month} предварительно заполнен.'
    return res


def get_backet_resulting_shares(df):
    res = {}
    for column in df.columns:
        tmp = column.split('_')
        if 'backet' in tmp:
            month = tmp[1]
            res[month] = df.groupby(column)[month].sum().to_dict()
    res = pd.DataFrame(res).T

    return res


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