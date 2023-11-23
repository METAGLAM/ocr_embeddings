import io
import os
from typing import List, Union, Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import unidecode

from src.tools import custom_typing as ct


# CSV load functions ==========================================================

def store_embeddings_weights_disk(
        labels: Union[pd.Series, pd.DataFrame], embeddings: np.ndarray,
        metadata_filename: str, checkpoint_filename: str) -> None:
    """
    This function takes a set of embeddings with a corresponding set of labels,
    and stores it into the path specified by 'metadata_filename' and
    'checkpoint_filename'. The former is to contain the labels, the latter is
    to contain the embeddings as weights. The purpose of the function is to
    store information as requested for Tensorboard visualisation purposes.

    Args:
        labels: a series or dataframe of labels
        embeddings: a numpy array containing n embeddings
        metadata_filename: full filename where to store the labels
        checkpoint_filename: full filename where to store the embeddings

    """
    labels.to_csv(metadata_filename, index=None, sep="\t")

    checkpoint = tf.train.Checkpoint(embedding=tf.Variable(embeddings))
    checkpoint.save(checkpoint_filename)


def create_embeddings_metadata(
        df: pd.DataFrame, columns: List[str], log_dir: str, filename: str) \
        -> None:
    """
    Generate a metadata file for the embeddings. It should be a tsv file with
    the columns associated to the embeddings and specified by parameter.

    Args:
        df: a pandas dataframe containing information related to the embeddings
        columns: the list of the dataframe columns to be stored
        log_dir: the destination directory
        filename: the name for the metadata file
    """
    with open(os.path.join(log_dir, filename), "w") as f:
        f.write('\t'.join(columns) + '\n')
        for i, row in df.iterrows():
            row_values = ''
            for c in columns:
                row_values += unidecode.unidecode(row[c]) + '\t'
            f.write(f"{row_values[:-1]}\n")


def load_muse_embedding(
        embeddings_path: str, max_words: Optional[int] = np.inf) \
        -> Dict[str, ct.Array]:
    """
    Loads the vectors of the MUSE multilingual model into a dict.

    Args:
        embeddings_path (str): path of the directory containing the file of
            the vectors.
        max_words (Optional[int]): maximum number of words to load. Set a
            value different from default only for quick prototyping during
            development.

    Returns:
        (dict): keys are words and values are vectors of size 300.

    Reference:
        https://github.com/facebookresearch/MUSE
    """
    embeddings_dict = {}
    with io.open(
            embeddings_path, 'r', encoding='utf-8', newline='\n',
            errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in embeddings_dict, 'word found twice'
            embeddings_dict[word] = vect

            if len(embeddings_dict) == max_words:
                break

    return embeddings_dict
