import os

import numpy as np
import spacy

from src.tools import custom_typing as ct


def execute_pipeline(
        name: str, texts: ct.StrList, enable: ct.StrList = None, **kwargs):
    """
    This function executes a spacy pipeline in a list of strings. Firstly, it
    loads the spacy model name. It is possible to loads only a set of pipeline
    steps. Then it applies each step to the text. Finally, it returns the
    transformed values.

    Args:
        name (str): model name to use.
        texts (ct.StrList): a list of strings to applies a spacy pipeline.
        enable (ct.StrList): a list of strings of the pipelines names to apply.

    Returns:
        a list with pipeline's processed steps.
    """
    if enable is None:
        enable = []

    nlp = spacy.load(name, enable=enable)

    return [doc for doc in nlp.pipe(texts, **kwargs)]


def compute_transformer_embeddings(
        name: str, texts: ct.StrList, **kwargs) -> ct.Array:
    """
    Given a model name and a list of strings, this function applies a spacy
    pipeline to generate the text embeddings.

    Args:
        name (str): model name to use.
        texts (ct.StrList): a list of strings to applies a spacy pipeline.

    Returns:
        (ct.Array): an array with embeddings.
    """
    enable = ['transformer']

    # Execute pipeline
    docs = execute_pipeline(name, texts, enable, **kwargs)
    # Get the embeddings
    embeddings = np.array([doc._.trf_data.tensors[-1] for doc in docs])

    return embeddings


def compute_word_embeddings(
        texts: ct.StrList, name: str, output_filename: str = None, **kwargs) \
        -> ct.Array:
    """
    Given a model name and a list of strings, this function applies a spacy
    pipeline to generate the word embeddings.

    Args:
        texts (ct.StrList): a list of strings to applies a spacy pipeline.
        name (str): model name to use.
        output_filename (str): the output file to load or store the embeddings.

    Returns:
        (ct.Array): an array with word embeddings.
    """
    if output_filename and os.path.exists(output_filename):
        word_embeddings = np.load(output_filename)
    else:
        enable = ['tok2vec']

        # Execute word embedding pipeline
        docs = execute_pipeline(name, texts, enable, **kwargs)
        # Get the word embeddings
        word_embeddings = np.array([doc.vector for doc in docs])

        # Store embedding to disk
        if output_filename:
            np.save(output_filename, word_embeddings)

    return word_embeddings
