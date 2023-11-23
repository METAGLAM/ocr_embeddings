import os
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from src.tools import custom_typing as ct


def generate_pipeline(pipeline_settings: dict) -> pipeline:
    """
    Given a settings of a HugginFace pipeline, this function generates the
    pipeline and returns it.

    Args:
        pipeline_settings (dict): settings of pipeline.

    Returns:
        (pipeline): The pipeline.
    """
    transformer_pipeline = pipeline(**pipeline_settings)
    return transformer_pipeline


def compute_transformer_embeddings(
        pipeline_settings, texts: ct.StrList, output_filepath: str = None,
        **_kwargs) -> ct.Array:
    """
    Given a configuration of HugginFace pipelines and a list of texts, this
    function generates the mean embedding (if it is not exists) of texts using
    a Transformer model. In addition, it stores the mean embedding into disk.

    model name and a list of strings, this function applies a spacy
    pipeline to generate the word embeddings.

    Args:
        pipeline_settings (dict): a Huggingface transformer pipeline
        texts (ct.StrList): a list of strings to compute the embeddings.
        output_filepath (str): the output file to load or store the embeddings.

    Returns:
        (ct.Array): an array with word embeddings.
    """
    if output_filepath and os.path.exists(output_filepath):
        mean_embedding = np.load(output_filepath)
    else:
        transformer_pipeline = generate_pipeline(pipeline_settings)
        # Generate embeddings for each token.
        embeddings = transformer_pipeline(texts)
        # Compute mean embedding of all texts.
        mean_embedding = np.mean(embeddings, axis=0)

        # Store embedding to disk
        if output_filepath:
            np.save(output_filepath, mean_embedding)

    return mean_embedding


def create_sentence_transformer(
        model_name: str, **kwargs) -> SentenceTransformer:
    """
    Given a model name, this function downloads the pre-trained model and
    creates a new instance of SentenceTransformer object.

    Args:
        model_name: the model name to load as a SentenceTransformer.

    Returns:
        (SentenceTransformer): a new SentenceTransformer model instance.
    """
    return SentenceTransformer(model_name, **kwargs)


def generate_average_embedding(
        model_name: str, texts: ct.StrList, output_filepath: str = None,
        model: Optional[SentenceTransformer] = None, **kwargs) -> ct.Array:
    """
    Generate embeddings using the SentenceTransformer class by using the model
    specified as argument. It loads the embeddings from the specified file if
    it exists, it generates the embeddings otherwise. In addition, it computes
    the average embeddings for each content.

    Args:
        model_name (str): the model name to load as a SentenceTransformer.
        texts (ct.StrList): a list of strings to compute the embeddings.
        output_filepath (str): the output file to load or store the embeddings.
        model (Optional[SentenceTransformer]): a SentenceTransformer model
            instance.

    Returns:
        (ct.Array): an array with word embeddings.
    """
    if output_filepath and os.path.exists(output_filepath):
        average_embedding = np.load(output_filepath)
    else:
        if model is None:
            model = create_sentence_transformer(model_name, **kwargs)
        average_embedding = np.mean(model.encode(texts), axis=0)
        np.save(output_filepath, average_embedding)

    return average_embedding
