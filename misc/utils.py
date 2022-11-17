import logging
import time
from datetime import timedelta
import importlib
from typing import Optional
import math


def timing(f):
    """
    Misura il tempo di esecuzione di una funzione.

    """

    def wrap(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        elapsed_time = (time.time() - start_time)

        logging.info(msg=f' {f.__name__} function took {str(timedelta(seconds=elapsed_time))}')

        return result

    return wrap


def suppress_tracking(f):
    """
    Disabilità il tracciamento del tracker prima dell'esecuzione della funzione.

    """

    def wrap(*args, **kwargs):
        tracker = args[0]
        tracker.enable_dataframe_warning_msg, tracker.dataframe_tracking, = False, False
        result = f(*args, **kwargs)
        tracker.dataframe_tracking, tracker.enable_dataframe_warning_msg = True, True

        return result

    return wrap


def convert(value: any, t: str):
    module = importlib.import_module("numpy")
    class_ = getattr(module, t)
    return class_(value)


def convert_to_int_no_decimal(value: float):
    """
    Converte un valore float in int solo se questo ha parte decimale pari a 0.
    In tutti i restanti casi la funzione restituisce il valore invariato.

    esempi:
        - 3.0 -> 3
        - 3.2 -> 3.2
        - 3 -> 3
        - '3.0' -> '3.0'

    :param value: valore da convertire.
    """
    if not isinstance(value, float):
        return value
    decimal, integer = math.modf(value)
    if decimal == 0:
        return int(integer)

    return value


def extract_used_features(code: str, features) -> Optional[list]:
    import re
    eq = code.split('=')
    first, second = eq[0], eq[1]

    used_df = first.split('[')[0]

    extracted_df_features = re.findall("\w*\[\'\w*\']", second)
    result = set()

    for extracted_df_feature in extracted_df_features:
        extracted_df_feature = re.split("\[|\]", extracted_df_feature)
        extracted_df_feature = tuple(filter(None, extracted_df_feature))
        extracted_df, extracted_feature = extracted_df_feature[0], extracted_df_feature[1].strip('\'')
        if extracted_feature in features and extracted_df == used_df:
            result.add(extracted_feature)

    return result


def invert_dictionary(dict1: dict):
    return {v: k for k, v in dict1.items()}


def keys_mapping(dict1: dict, dict2: dict):
    result = dict()
    dict2 = {v: k for k, v in dict2.items()}

    for key1, value in dict1.items():
        key2 = dict2.get(value, None)
        if key2:
            result[key1] = key2

    return result
