import json
import os
from typing import Dict, List, Union

import numpy as np
import pandas as pd


class OcrManagement:
    """
    This class takes care or managing some tasks needed before processing OCRs
    and after.

    Attributes:
        _path_bloc: path where the files to process are stored.

        _revistes: (optional) list of magazines to process given by the user, if
            not provided, all the magazines in the folder will be processed.

        _df_path: filepath of the statistics dataframe. To read previous
            executions and append new ones.

        _all_words_path: filepath of the general dictionary containing
            words of all files processed. To read previous executions and
            append new ones.

        _processed_text_path: path of the folder containing processed files.

        _processed_lines_path: filepath of the json file dictionary containing
            all processed lines with misspelled words.

        _columns: list of columns that will be present in the statistics df.

    Methods:
        list_docs_to_process(): List the files to process, compares them with
            the files that have already been processed in previous executions,
            and returns a list of files to process in this execution.

        write_processed_ocr(): Save the processed OCR file as a .txt in a
            processed files' folder.

        update_words_json(): Save a json containing the _all_words dictionary.

        compute_statistics(): Compute statistics, update the _df dataframe, and
            save it as a csv.

    """

    def __init__(self, settings: Dict):
        """
        This is the constructor method of the OcrManagement class. It
        initializes the class attributes depending on the provided settings.

        Args:
            settings: load settings that define the path of the files,
                the results path and the columns of the statistics dataframe.
        """
        path_publicacions = settings["transform"]["path_publicacions"]
        bloc = settings["transform"]["bloc"]
        self._path_bloc = os.path.join(path_publicacions, bloc)
        self._revistes = settings["transform"].get("revistes", None)

        results_path = settings["save"]["results_path"]
        file_settings = settings["save"]["file"]

        self._df_path = os.path.join(results_path, file_settings[
            "statistics"]["filepath"].replace("{bloc}", bloc))
        self._all_words_path = os.path.join(results_path, file_settings[
            "all_words"]["filepath"].replace("{bloc}", bloc))
        self._processed_text_path = os.path.join(results_path, file_settings[
            "processed_text"]["filepath"].replace("{bloc}", bloc))
        self._processed_lines_path = os.path.join(results_path, file_settings[
            "processed_lines"]["filepath"].replace("{bloc}", bloc))

        self._columns = settings["transform"]["columns"]

        self._check_corrected_ocr_folder()

    def _check_previous_execution(self, file: str) \
            -> Union[pd.DataFrame, Dict, List]:
        """
        Search for previous results file in case the execution stopped
        halfway, and we want it to resume where it stopped
        Returns:
            df: statistics dataframe computed in previous executions.
            all_words: general dictionary containing words of all files
                processed in previous executions.
            processed_lines: list containing processed lines with a misspelled
                word.
        """
        if file == 'df':
            if os.path.exists(self._df_path):
                df = pd.read_csv(self._df_path)
                df['revista'] = df['revista'].astype(str)
                return df
            else:
                return pd.DataFrame(columns=self._columns)

        if file == 'all_words':
            if os.path.exists(self._all_words_path):
                all_words = json.load(open(self._all_words_path))
                return all_words
            else:
                return {}

        if file == 'processed_lines':
            if os.path.exists(self._processed_lines_path):
                processed_lines = json.load(open(self._processed_lines_path))
                return processed_lines
            else:
                return []

        raise Exception(f"Invalid value {file}, allowed values are: df, "
                        f"all_words, and processed_lines.")

    def _check_corrected_ocr_folder(self, revista: str = None) -> None:
        """
        Check if the folder to save corrected OCRs exists, and if it doesn't,
        create it.
        Args:
            revista: optional, name of the magazine and folder that will
                contain its documents.
        """
        path = [self._processed_text_path, revista]
        path = [level for level in path if level]
        if not os.path.exists(os.path.join(*path)):
            os.mkdir(os.path.join(*path))

    def list_docs_to_process(self) -> List[tuple[str, str]]:
        """
        Lists all the files in the file path, compares them with the list of
        files that have already been processed in previous executions,
        and returns a list of files to process.
        Returns:
            docs_to_process: a list with a tuple of (magazine, file) for each
                file that has to be processed.
        """
        docs_to_process = []

        revista_list = os.listdir(self._path_bloc)
        if self._revistes:
            revista_list = [revista for revista in revista_list if revista in
                            self._revistes]
        revista_list.sort()

        df = self._check_previous_execution('df')

        for revista in revista_list:
            if revista[0] != '.':
                docs_list = os.listdir(os.path.join(self._path_bloc, revista))
                docs_list.sort()

                for doc in docs_list:
                    if (doc.endswith('.txt')) and not \
                            ((revista in df['revista'].values) and
                             (doc in df[df['revista']
                                     .isin([revista])]['publicacio'].values)):
                        docs_to_process.append((revista, doc))

        return docs_to_process

    def write_processed_lines(
            self, misspelled_words: bool, line: Union[str, Dict],
            processed_line: str, revista: str, file: str, line_num: int,
            lang: str) -> None:
        """
        If there are misspelled words in the line save them in the json file.
        Args:
            misspelled_words: boolean stating whether the line contains any 
                originally misspelled word.
            line: original line.
            processed_line: processed line.
            revista: magazine to which the line belongs.
            file: file to which the line belongs.
            line_num: line number in the file.
            lang: identified language of the line.
        """
        processed_lines = self._check_previous_execution('processed_lines')

        if type(line) == dict:
            raw_line = line['raw']
            preprocessed_line = line['preprocessed']
        else:
            raw_line = line
            preprocessed_line = ''

        if misspelled_words:
            processed_lines.append({
                "raw_line": raw_line,
                "preprocessed_line": preprocessed_line,
                "processed_line": processed_line,
                "revista": revista,
                "file": file,
                "line_num": line_num,
                "lang": lang
            })
            json.dump(processed_lines,
                      open(self._processed_lines_path, "w"))
        """
        processed_lines.append({
            "raw_line": raw_line,
            "preprocessed_line": preprocessed_line,
            "processed_line": processed_line,
            "revista": revista,
            "file": file,
            "line_num": line_num,
            "lang": lang
        })
        json.dump(processed_lines,
                  open(self._processed_lines_path, "w", encoding='utf-8'))
        """

    def write_processed_ocr(
            self, revista: str, file: str, processed_text: List[str]) -> None:
        """
        Save the processed OCR file as a .txt in a processed files' folder.
        Args:
            revista: name of the magazine the file belongs to.
            file: name of the file.
            processed_text: text that will be written.
        """
        self._check_corrected_ocr_folder(revista)

        open(os.path.join(self._processed_text_path, revista, file[:-4] +
                          '_processed.txt'), 'w').writelines(processed_text)

    def update_words_json(self, file_words: Dict, revista: str, file: str) \
            -> None:
        """
        Integrate the words file dictionary of the given file with the
        general dictionary containing words of all files and saves a checkpoint.
        Args:
            file_words: dictionary with the existing and non-existing words
                of the file, classified as well per language.
            revista: name of the magazine the file belongs to.
            file: name of the file.
        """
        all_words = self._check_previous_execution('all_words')
        if revista not in all_words:
            all_words[revista] = {}

        all_words[revista][file[:-4]] = file_words

        json.dump(all_words, open(self._all_words_path, "w"))

    def compute_statistics(self, file_words: Dict, revista: str, file: str) \
            -> None:
        """
        Extracts count and percentage of existing and non-existing words in
        the given file and language.
        Updates df dataframe with data of the file and saves a checkpoint.

        Args:
            file_words: dictionary with the existing and non-existing words
                of the file, classified as well per language.
            revista: name of the magazine the file belongs to.
            file: name of the file.
        """
        count_existing = {}
        count_lang = {}
        for lang, value in file_words.items():
            count_existing[lang] = len(value["existing"])
            count_lang[lang] = (len(value["existing"]) +
                                len(value["non_existing"]))

        row = {
            "revista": revista,
            "publicacio": file,
            "idiomes": [],
            "n_words": sum(count_lang.values()),
            "total_existing":
                round(np.divide(sum(count_existing.values()),
                                sum(count_lang.values())) * 100, 2)
        }

        for lang, count in count_lang.items():
            col_perc = lang + "_perc"
            percentage = round(
                np.divide(count, sum(count_lang.values())) * 100,
                2)
            row[col_perc] = percentage
            if (percentage > 5) and (lang != "non_identified"):
                row["idiomes"].append(lang)

            col_existing = lang + "_existing"
            percentage = round(np.divide(count_existing[lang], count) * 100, 2)
            row[col_existing] = percentage

        df = self._check_previous_execution('df')

        df.loc[len(df.index)] = row
        df.to_csv(self._df_path, index=False)
