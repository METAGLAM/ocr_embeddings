import json
import os
from typing import Optional

import pandas as pd

from src.tools import custom_typing as ct
from src.tools.startup import logger


def read_json(filepath: str, **kwargs) -> dict:
    """
    Given a file path, this function reads the file as json and gets the
    content as a Python dict.

    Args:
        filepath (str): a json's file path.

    Returns:
        (dict): a dict with json content.
    """
    try:
        with open(filepath, **kwargs) as json_file:
            content = json.load(json_file)
    except FileNotFoundError as _:
        logger.warning(f"File path '{filepath}' does not exists.")
        content = {}

    return content


def read_processed_ocrs(
        languages: ct.StrList, files: ct.StrList) -> pd.DataFrame:
    """
    Given a list of languages and the files paths, this function reads the
    files and filters the content for the languages list.

    Args:
        languages (ct.StrList): a list of languages to filter.
        files (ct.StrList): a list of files to read.

    Returns:
        (pd.DataFrame): a pd.DataFrame where each row corresponds a document
        and each column corresponds to a language text plus
        additional metadata.
    """
    # Read json data
    post_processed_ocrs = {}
    for file in files:
        content = read_json(file)
        post_processed_ocrs.update(content)
    # Create a dataframe with the data
    dataframe_rows = []
    for magazine in post_processed_ocrs:
        for publication_id in post_processed_ocrs[magazine]:
            row = {
                'doc_id': f'{magazine}_{publication_id}',
                # TODO: adding metadata
                'title': magazine,
            }
            for lang in languages:
                if lang in post_processed_ocrs[magazine][publication_id]:
                    words = post_processed_ocrs[magazine][publication_id][
                        lang]['existing']
                    row[f'{lang}_text'] = ' '.join(words)
            dataframe_rows.append(row)

    return pd.DataFrame(dataframe_rows)


def read_processed_ocr_full_text(
        folders: ct.StrList, encoding: Optional[str] = 'utf-8', **kwargs) \
        -> pd.DataFrame:
    """
    Given a list of folders, this function gets all files for each magazine
    and stores it to a dataframe.
    
    Args:
        folders (ct.StrList): a list of folders to read the content magazines
            and files.
        encoding (Optional[str]): the encoding to use to read file. Default
            value is 'utf-8'.

    Returns:
        (pd.DataFrame): a pd.DataFrame where each row corresponds a document.
            In addition, the column "text" has the full text of the document.
    """
    data = []
    for folder in folders:
        for title in os.listdir(folder):
            magazine_path = os.path.join(folder, title)
            for filename in os.listdir(magazine_path):
                # If processed remove from filename.
                if 'processed' in filename:
                    filename_split = filename.split('_')
                    if len(filename_split) == 2:
                        doc_id = f"{title}_{filename.split('_')[0]}"
                    else:
                        doc_id = \
                            f"{title}_{'_'.join(filename.split('_')[:-1])}"
                # Otherwise, it uses the filename directly.
                else:
                    filename_split = filename.split('.')
                    doc_id = f'{title}_{filename_split[0]}'

                file_path = os.path.join(magazine_path, filename)
                try:
                    with open(file_path, 'r', encoding=encoding, **kwargs) \
                            as file:
                        content = file.read()
                except UnicodeError:
                    with open(file_path, 'r', encoding='latin-1', **kwargs) \
                            as file:
                        content = file.read()

                data.append({
                    'doc_id': doc_id,
                    'title': title,
                    'text': content
                })

    df = pd.DataFrame(data)
    df['text'] = df.text.str.replace('\n', ' ')

    return df
