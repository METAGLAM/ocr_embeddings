import re

import numpy as np
import unidecode
from gensim.parsing import preprocessing as pproc
from sklearn.manifold import TSNE


def clean_string(string: str) -> str:
    """
    Applies cleaning operations to a string, including decoding to unicode,
    lower-casing, abbreviation resolution (U.S.A. -> usa), punctuation removal
    and blank space stripping.

    Args:
        string (str): string that you want to clean.

    Returns:
        (str): The clean string
    """
    string = unidecode.unidecode(string)
    string = string.lower()
    abbreviations = re.findall(r'(?:[a-z]\.)+', string)
    for abbr in abbreviations:
        string = string.replace(abbr, abbr.replace('.', ''))
    string = pproc.strip_punctuation(string)
    string = string.strip()

    return string


def tsne_transformation(data: np.ndarray, **kwargs) -> np.ndarray:
    """
    Given an array, this function applies a dimensionality reduction
    transformation.

    Args:
        data (np.ndarray): an array to transform.
        **kwargs (dict): arguments of TSNE.

    Returns:
        (np.ndarray): an array with the dimensionality reduction.
    """
    tsne = TSNE(**kwargs)
    return tsne.fit_transform(data)
