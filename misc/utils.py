import importlib
import math
import re
from typing import Optional, Dict, List


def convert(value: any, t: str) -> any:
    """
    Converts a value to the specified type.

    :param value: The value to be converted.
    :param t: The target type.
    :return: The converted value.
    """

    module = importlib.import_module("numpy")
    class_ = getattr(module, t)
    return class_(value)


def convert_to_int_no_decimal(value: float) -> int | float:
    """
    Converts a float value to an int if the decimal part is 0.
    Returns the value unchanged in all other cases.

    Examples:
        - 3.0 -> 3
        - 3.2 -> 3.2
        - 3 -> 3
        - '3.0' -> '3.0'

    :param value: The value to convert.
    :return: The converted value.
    """
    if not isinstance(value, float):
        return value
    decimal, integer = math.modf(value)
    if decimal == 0:
        return int(integer)

    return value


def extract_used_features(code: str, features: List[str]) -> Optional[list]:
    """
    Extracts used features from code.

    :param code: The code string.
    :param features: The features to search for.
    :return: A list of extracted features.
    """

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


def invert_dict(dictionary: Dict[int, any]) -> Dict[any, int]:
    """
    Inverts a dictionary by swapping keys and values.

    :param dictionary: The input dictionary.
    :return: The inverted dictionary.
    """

    inverted_dict = {}
    for key, value in dictionary.items():
        inverted_dict.setdefault(value, set()).add(key)
    return inverted_dict


def keys_mapping(dict1: Dict[int, any], dict2: Dict[any, any]) -> Dict[int, any]:
    """
    Maps keys from dict1 to keys from dict2.

    :param dict1: The first dictionary.
    :param dict2: The second dictionary.
    :return: A dictionary containing the mapping between keys from dict1 and dict2.
    """

    result = dict()

    dict2 = invert_dict(dict2)

    for key1, value in dict1.items():
        key2 = dict2.get(value, None)
        if key2 is None:
            continue
        if key1 in key2:
            result[key1] = key1
        else:
            result[key1] = key1

    return result
