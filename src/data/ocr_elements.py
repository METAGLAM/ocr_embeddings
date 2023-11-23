import copy
import os
import re
from typing import List, Dict, Union

import chardet
from Levenshtein import distance
# library to detect languages
from langdetect import detect_langs, LangDetectException
# library to spellcheck words statistically
from symspellpy import Verbosity

from src.data.resource_loader import ResourceLoader


class OcrText:
    """
    This class represents an OCR text.

    Attributes:
        _path_bloc: path where the file is stored.

        _revista: name of the magazine the file belongs to.

        _file: name of the file to process.

        _raw_text: unprocessed text obtained after reading the content of the
            file.

        _data_file_lines: list containing the lines of the text.

    Methods:
        preprocess_text(): preprocess and clean the raw text.

        extract_lines(): return a list containing the lines of the text.
    """

    def __init__(self, settings: Dict, doc_to_process: tuple,
                 vocabularies: Dict):
        """
        This is the constructor method od the OcrText class. It initializes
        the class attributes depending on the provided settings and the
        doc_to_process tuple.

        Args:
            settings: load settings that define the path of the file.
            doc_to_process: tuple containing name of the magazine and file.
            vocabularies: dictionary with a key for every language containing a
                set with all the words of the vocabulary and a set with all the
                stop words.
        """

        path_publicacions = settings["transform"]["path_publicacions"]
        bloc = settings["transform"]["bloc"]
        self._path_bloc = os.path.join(path_publicacions, bloc)
        self._revista, self._file = doc_to_process

        vocabulary = vocabularies['non_identified']["vocabulary"]
        stop_words = vocabularies['non_identified']["stop_words"]
        self._vocab_stopw = vocabulary.union(stop_words)

        self._raw_text = self._read_file()

        self._data_file_lines = [line for line in self._raw_text.split("\n")]
        if settings["transform"]["preprocess"]:
            self._preprocess_text()
        self._paragraph_settings = settings["transform"]["paragraph"]
        self._data_file_paragraphs = self._join_lines_into_paragraphs()

    def _read_file(self) -> str:
        """
        Given the name of the magazine and the name of the file,
        this method opens the file, reads it, and returns its content.

        Returns:
            data_file: string containing the content of the file
        """

        filepath = os.path.join(self._path_bloc, self._revista, self._file)

        with open(filepath, 'rb') as file_opener:
            raw_data = file_opener.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']

        try:
            with open(filepath, encoding=encoding) as file_opener:
                data_file = file_opener.read()
        except UnicodeDecodeError:
            with open(filepath, encoding='latin-1') as file_opener:
                data_file = file_opener.read()

        """
        with open(filepath, encoding='utf-8') as file_opener:
            data_file = file_opener.read()
        """

        return data_file

    def _preprocess_text(self) -> None:
        """
        This method preprocesses and cleans the raw text read from the files.
        The steps applied are the following:
            1. Remove weird characters and punctuation signs.
            2. Separate text into lines, removes whitespace at the beginning or
                end of the line.
            3. Remove single consonants.

        Creates:
            data_file_lines: List[str] a list where each element is a line of
            the text after being preprocessed.
        """

        # remove characters not present in the list
        chars_to_keep = r"[^a-z 0-9 A-Z áéíóúàèòâêîôûïü ÁÉÍÓÚÀÈÒÂÊÎÔÛÏÜ ñ ç " \
                        r"' \- — - , . · ; : ? ! \n]"
        data_file = re.sub(chars_to_keep, "", self._raw_text)

        # remove weird char \x0c
        data_file = data_file.replace("\x0c", "")

        # separates text into lines, removes whitespace at the beginning or end
        # of the line, replaces hyphens coded with different chars
        data_file_lines = \
            [line_.replace("—", "-").replace("-", "-") for line_ in
             [" ".join(line.split()) for line in data_file.split("\n")]]

        # remove single consonants
        """data_file_lines = \
            [" ".join([word for word in line.split() if (len(word) > 1) or
                       (word.lower() not in
                        ['q', 'w', 'r', 't', 'p', 's', 'd', 'f', 'g', 'h', 'j',
                         'k', 'l', 'ñ', 'z', 'x', 'c', 'v', 'b', 'n', 'm',
                         'ç'])])
             for line in data_file_lines]"""

        # Delete empy lines, lines with all chars upper case (titles), numbers
        # (page nums, phone nums, etc.), non-alphabetic chars (typos, scan
        # noise, etc.)
        self._data_file_lines = \
            [{'raw': raw_line, 'preprocessed': preprocessed_line}
             for raw_line, preprocessed_line
             in list(zip(self._data_file_lines, data_file_lines))
             if (any(c.isalpha() for c in preprocessed_line))
             and (preprocessed_line.upper() != preprocessed_line)]

    def extract_lines(self) -> List[Union[str, Dict]]:
        """
        Getter method of the list of lines of the text.
        Returns:
            data_file_lines: list containing the lines of the text.
        """
        return self._data_file_lines

    def _join_lines_into_paragraphs(self) -> List[Union[Dict, str]]:
        """
        Joins the lines of the text into paragraphs following heuristics.
        Returns:
            List of paragraphs.
        """
        if type(self._data_file_lines[0]) == dict:
            data_file_lines = \
                [line['preprocessed'] for line in self._data_file_lines]
        else:
            data_file_lines = [line for line in self._data_file_lines if
                               any(c.isalpha() for c in line)]

        line_parag = [i for i in range(len(data_file_lines))]
        len_parag = {}
        for i, line in enumerate(data_file_lines):
            len_parag[i] = len(line.split())

        for i, raw_line in enumerate(data_file_lines):

            line = self._process_line(raw_line)

            # cases when the following line joins the current line paragraph.
            if (i < len(data_file_lines) - 1) and \
                    self._join_lines_heuristic(line, len_parag[line_parag[i]]):
                len_parag[line_parag[i]] += len_parag.pop(line_parag[i + 1])
                line_parag[i + 1] = line_parag[i]

            # if the first word of the line does not exist in the vocabulary the
            # current line will join the previous paragraph.
            if (i > 0) and (line_parag[i - 1] != line_parag[i]) and \
                    (line.split()[0].lower() not in self._vocab_stopw):

                len_parag[line_parag[i - 1]] += len_parag.pop(line_parag[i])

                if i < len(line_parag) - 1:
                    if line_parag[i + 1] == line_parag[i]:
                        line_parag[i + 1] = line_parag[i - 1]
                line_parag[i] = line_parag[i - 1]

        # join lines in paragraphs
        parag = {el: '' for el in sorted(list(set(line_parag)))}
        for line, par in enumerate(line_parag):
            if data_file_lines[line][-1] == '-':
                parag[par] += (data_file_lines[line][:-1])
            else:
                parag[par] += (data_file_lines[line] + ' ')

        paragraphs = [par.strip() for par in parag.values()]

        if type(self._data_file_lines[0]) == dict:
            preproc_paragraphs = paragraphs
            raw_data_file_lines = \
                [line['raw'] for line in self._data_file_lines]

            # join raw lines in paragraphs
            parag = {el: '' for el in sorted(list(set(line_parag)))}
            for line, par in enumerate(line_parag):
                if raw_data_file_lines[line][-1] == '-':
                    parag[par] += (raw_data_file_lines[line][:-1])
                else:
                    parag[par] += (raw_data_file_lines[line] + ' ')
            raw_paragraphs = [par.strip() for par in parag.values()]

            paragraphs = [{'raw': raw_par, 'preprocessed': preprocessed_par}
                          for raw_par, preprocessed_par
                          in list(zip(raw_paragraphs, preproc_paragraphs))]

        """random.seed(123)
        selected_index = random.randint(0, len(paragraphs)-1)
        selected_paragraph = [paragraphs[selected_index]]
        """
        return paragraphs

    @staticmethod
    def _process_line(raw_line: str):
        # remove punctuation signs
        line = raw_line.replace(',', '').replace(':', '') \
            .replace('?', '').replace(';', '').replace('!', '')
        return line

    def _join_lines_heuristic(self, current_line: str, len_current_parag: int) \
            -> bool:
        """
        Heuristics that define in which cases we will be joining two lines
        into the same paragraph.
        Args:
            current_line (str): text of the current line.
            len_current_parag (int): length of the current line paragraph.

        Returns:
            (bool): boolean defining whether the following line will be
            joined to the current line paragraph.
        """

        # if last word of the line ends with a hyphen, the current line will
        # join the following
        if current_line[-1] == '-':
            return True

        # if the number of words in the paragraph is inferior to the hard
        # minimum, lines will join
        hard_min_len = len_current_parag < self._paragraph_settings['hard_min']
        if hard_min_len:
            return True

        # if the length of the line is over the maximum allowed, lines won't
        # join
        max_len = len_current_parag > self._paragraph_settings['max']
        if max_len:
            return False

        # if the last word of the line does not exist in the vocabulary, it
        # is not capitalized (name, place, etc.), and it contains at least
        # one alphabetical character
        # AND it does not end with an end-of-paragraph character,
        # lines will join
        last_word = current_line.split()[-1].replace('.', '')
        word_not_existing = (last_word.lower() not in self._vocab_stopw) and \
                            (last_word != last_word.capitalize()) and \
                            (any(c.isalpha() for c in last_word))
        end_of_parag_characters = ['.', '...', '?', '!']
        end_characters = current_line[-1] in end_of_parag_characters

        if word_not_existing and not end_characters:
            return True

        # if the length of the line is not over the soft minimum
        # AND the last word of the line does not end with an end-of-paragraph
        # character, lines will join
        soft_min_len = len_current_parag < self._paragraph_settings['soft_min']

        if soft_min_len and not end_characters:
            return True

        # In any other case, the current line won't join the following
        return False

    def extract_paragraphs(self) -> List[Union[str, Dict]]:
        """
        Getter method of the paragraphs of lines of the text.
        Returns:
            data_file_paragraphs: list containing the paragraphs of the text.
        """
        return self._data_file_paragraphs


class OcrLine:
    """
    This class represents a line of the OCR text.

    Attributes:
        _line: line of the text to process.

        _vocabularies: dictionary with a key for every language containing a
            set with all the words of the vocabulary and a set with all the
            stop words.

        file_words: dictionary containing all the words in the file, classified
            by language, and by whether they exist in the language dictionary or
            not.

        lang: language of the line.

        line_words: list containing the words of the line.
    """

    def __init__(self, line: Union[str, Dict], vocabularies: Dict,
                 file_words: Dict):
        """
        This is the constructor method od the OcrLine class. It initializes
        the class attributes depending on the provided attributes.

        Args:
            line: line of the text to process.
            vocabularies: dictionary with a key for every language containing a
                set with all the words of the vocabulary and a set with all the
                stop words.
            file_words: dictionary containing all the words in the file,
                classified by language, and by whether they exist in the
                language dictionary or not.
        """
        if type(line) == dict:
            self._line = line['preprocessed']
        else:
            self._line = line
        self._vocabularies = vocabularies
        self.file_words = file_words

        self.lang = self._detect_language()
        self._add_lang_to_file_words()
        self.line_words = self._extract_words()

    def _detect_language(self) -> str:
        """
        This method uses langdetect library to identify the line's language.
        Returns:
            if the language of the line was identified, the language code of
            the line, "non_identified" otherwise.
        """
        try:
            langs = detect_langs(self._line)
            for lang in langs:
                lang = str(lang)[:2]
                if lang in self._vocabularies.keys():
                    return lang

            return "non_identified"

        except LangDetectException:
            return "non_identified"

    def _add_lang_to_file_words(self):
        """
        Add the language of the line to the file words dictionary.
        """
        if self.lang not in self.file_words:
            self.file_words[self.lang] = {"existing": [], "non_existing": []}

    def _extract_words(self):
        """
        This method splits the punctuation signs from the words and returns a
        list of words of the line.
        Returns:
            word_list: list of words of the line
        """
        line = str(self._line + ' ').replace(". ", " . ").replace(", ", " , ") \
            .replace(": ", " : ").replace("; ", " ; ") \
            .replace("? ", " ? ").replace("! ", " ! ").replace("  ", " ")

        word_list = [word for word in line.split(" ")[:-1] if word != '']
        return word_list


class OcrWord:
    """
    This class represents a single word of the OCR text.

    Attributes:
        _word: word to process.

        _line: line to which the word to be processed belongs represented as a
            word list.

        _lang: language of the line to which the word to be processed belongs.

        _word_position: position of the word in reference to the line
            represented as a word list.

        _stop_words: set of stop words of the specified language, extracted
            from the vocabularies' dictionary.

        _vocabulary: set of existing words of the specified language, extracted
            from the vocabularies' dictionary.

        _sym_spell: loaded SymSpell instance of the specified language.

        _fill_mask: loaded BERT fill_mask instance of the specified language.

        apostrophe_words: dictionary containing, in case the word contains an
            apostrophe, the position of the word, the position of the
            apostrophe pronoun, and the apostrophe pronoun.

        hyphen_words: dictionary containing, in case the word contains a hyphen,
            the position of the word and the hyphen pronoun.

        file_words: dictionary containing all the words in the file, classified
            by language, and by whether they exist in the language dictionary or
            not.

        _apostrophe: list that define the accepted pronouns for apostrophes.

        _hyphen: list that define the accepted pronouns for hyphens.

    Methods:
        process_word(): Check if the word exist in the dictionary, and if it
            does not, execute the spellchecker method to try to correct it.
            Then, check again if the resulting word exist in the dictionary.
            Exclude words that do not contain any alphabetic character and
            stop words.
            Remove apostrophes and hyphens of the words that are written in
            languages that use these punctuation marks and add them back again
            after finishing the existence check.
            Remove capital letters of the word, except for the first character.
            Either if the word exists or not in the dictionary, update the
            file_words dictionary and return the processed word to be added
            to the processed text file.

    """

    def __init__(self, word: str, line: List[str], lang: str,
                 word_position: int, resources: ResourceLoader, settings: Dict,
                 apostrophe_words: Dict = {}, hyphen_words: Dict = {},
                 file_words: Dict = {}):
        """
        This is the constructor method od the OcrWord class. It initializes
        the class attributes depending on the provided arguments.

        Args:
            word: word to process.
            line: line to which the word to be processed belongs represented
                as a word list.
            lang: language of the line to which the word to be processed
                belongs.
            word_position: position of the word in reference to the line
                represented as a word list.
            resources: resources loaded by the ResourceLoader containing the
                vocabularies, sym_spell instances and fill_mask instances.
            settings: dict that define the accepted pronouns for apostrophes and
                hyphens.
            apostrophe_words: dict containing, in case the word contains an
                apostrophe, the position of the word, the position of the
                apostrophe pronoun, and the apostrophe pronoun.
            hyphen_words: dict containing, in case the word contains a hyphen,
                the position of the word and the hyphen pronoun.
            file_words: dictionary containing all the words in the file,
                classified by language, and by whether they exist in the
                language dictionary or not.
        """
        self._word = word
        self._line = line
        self._lang = lang
        self._word_position = word_position

        self._stop_words = resources.vocabularies[self._lang]["stop_words"]
        self._vocabulary = resources.vocabularies[self._lang]["vocabulary"]

        self.apostrophe_words = apostrophe_words
        self.hyphen_words = hyphen_words
        self.file_words = file_words

        self._apostrophe = settings["transform"]["apostrophe"]
        self._hyphen = settings["transform"]["hyphen"]

        self._use_symspell = settings["transform"]["steps"]["use_symspell"]
        self._use_bert = settings["transform"]["steps"]["use_bert"]

        if self._use_symspell:
            self._sym_spell = resources.sym_spell[self._lang] \
                if self._lang != "non_identified" else None
        if self._use_bert:
            self._fill_mask = resources.fill_mask[self._lang] \
                if self._lang != "non_identified" else None

        self.originally_misspellen = False

    def process_word(self) -> str:
        """
        This method checks if the word exists in the dictionary, and if it
        does not, it executes the spellchecker method to try to correct it.
        Then it checks again if the resulting word exists in the dictionary.
        It excludes words that do not contain any alphabetic character and
        stop words.
        It removes apostrophes and hyphens of the words that are written in
        languages that use these punctuation marks and adds them back again
        after finishing the existence check.
        It removes capital letters of the word, except for the first character.

        Returns:
            word: processed word.
        """
        if (any(c.isalpha() for c in self._word)) and \
                (self._word not in self._stop_words) and \
                (self._word.lower() not in self._stop_words):

            # remove apostrophes from words
            self._process_apostrophe()
            # remove hyphens from words
            self._process_hyphens()

            # consider correct Capitalized words (names, cities, ...)
            if self._word == self._word.capitalize():
                self.file_words[self._lang]["existing"].append(self._word)

            else:
                # only keep 1st capital letter of a word
                self._word = self._word.capitalize() if self._word[0].isupper() \
                    else (self._word.lower())

                # check word existence
                if ((self._word in self._vocabulary) or
                        (self._word.lower() in self._vocabulary) or
                        (self._word.capitalize() in self._vocabulary) or
                        (self._lang == "non_identified")):
                    self._check_word_existence(self._word)

                else:
                    if self._word != '':

                        self.originally_misspellen = True

                        # spellchecker
                        self.spellchecker()

                        # the spellchecker returned a single word
                        if type(self._word) == str:
                            self._check_word_existence(self._word)

                        # the spellchecker split the word in two
                        elif type(self._word) == list:
                            for w in self._word:
                                self._check_word_existence(w)
                            self._word = " ".join(self._word)

            # add back apostrophes to words
            self._word = self._add_back_apostrophe(self._word)
            # add back hyphens to words
            self._word = self._add_back_hyphen(self._word)

        return self._word

    def _process_apostrophe(self) -> None:
        """
        Find, remove, and record apostrophes from words in Catalan and French
        languages.
        """
        if (self._lang in ['ca', 'fr']) and ("'" in self._word):

            w = self._word.split("'")

            if len(w[0]) <= len("'".join(w[1:])):
                ap = w[0] + "'"
                short_word = "'".join(w[1:])
                position = "beginning"

            else:
                ap = "'" + w[1]
                short_word = w[0]
                position = "ending"

            if ap.lower() in self._apostrophe[self._lang]:
                self.apostrophe_words[self._word_position] = {position: ap}
                self._word = short_word

    def _add_back_apostrophe(self, word) -> str:
        """
        Add back the apostrophe to the word
        Args:
            word: initial state of the word, with or without an apostrophe.

        Returns:
            word: in case it had one, word without the apostrophe, otherwise,
            the original word.
        """
        if self._word_position in self.apostrophe_words:
            if 'beginning' in self.apostrophe_words[self._word_position]:
                return str(
                    self.apostrophe_words[self._word_position]['beginning'] +
                    word)
            if 'ending' in self.apostrophe_words[self._word_position]:
                return str(word +
                           self.apostrophe_words[self._word_position][
                               'ending'])
        else:
            return word

    def _process_hyphens(self) -> None:
        """
        Find, remove, and record hyphens from words in Catalan language.
        """
        if (self._lang == 'ca') and ("-" in self._word):

            w = self._word.split("-")
            hy = "-" + w[1]
            short_word = w[0]

            if (hy in self._hyphen) and (len(short_word) > 0):
                self.hyphen_words[self._word_position] = hy
                self._word = short_word

    def _add_back_hyphen(self, word) -> str:
        """
        Add back the hyphen to the word.
        Args:
            word: initial state of the word, with or without a hyphen.

        Returns:
            word: in case it had one, word without the hyphen, otherwise,
            the original word.
        """
        if self._word_position in self.hyphen_words:
            return str(word + self.hyphen_words[self._word_position])
        else:
            return word

    def _check_word_existence(self, word) -> None:
        """
        This method checks if the word is not a stop word and whether it is
        contained in the vocabulary of the word language, if it is,
        the dictionary file_words get updated.

        Args:
            word: word to check existence.
        """

        if (word not in self._stop_words) and (
                word.lower() not in self._stop_words):
            if word in self._vocabulary:
                self.file_words[self._lang]["existing"].append(word)
            elif word.lower() in self._vocabulary:
                self.file_words[self._lang]["existing"].append(word.lower())
            elif word.capitalize() in self._vocabulary:
                self.file_words[self._lang]["existing"].append(
                    word.capitalize())
            else:
                self.file_words[self._lang]["non_existing"].append(word)

    def spellchecker(self) -> None:
        """
        This method corrects misspelled words by following 4 different
        approaches:
        1st. Use sym_spell to obtain candidates to correct the misspelled word.
            In case there is only one candidate (with Levenshtein distance > 0),
            that word will be returned.
        2nd. If sym_spell returns more than one candidate, it will call BERT
            with targets (being sym_spell candidates the targets) to make
            suggestions based on the context of the line. If there is any
            suggestion with a score higher than 0.0001, it will return the one
            with the highest score.
        3rd. If none of the suggestions have a score > 0.0001, then BERT without
            targets will be called to get top 30 predictions and check if any
            matches sym_spell output. (This is done because in some occasions
            executing RoBERTa with targets breaks the target words by tokens and
            prevents using the full word). If there is any match, it will be
            returned.
        4th. If there isn't any match, the last option is to check if the word
            is a composition of two words, using sym_spell compose, it does not
            consider Capitalized words in case they are names. If the misspelled
            word is a composition of two words, both will be returned as a list,
            if it is not, the original word will be returned.
        """

        if self._use_symspell:
            # use sym_spell to obtain candidates to correct the misspelled word
            suggestions = self._sym_spell.lookup(
                self._word, Verbosity.ALL, max_edit_distance=2,
                include_unknown=True)

            if len(suggestions) == 1:
                self._word = suggestions[0].term
                if suggestions[0].distance > 0:
                    return

        if self._use_bert:
            # prepare masked line to use BERT
            masked_word_list = copy.deepcopy(self._line)
            mask_token = self._fill_mask.tokenizer.mask_token

            # add back the apostrophe to the word and replace it by the mask
            # token
            masked_word = self._add_back_apostrophe(mask_token)
            # add back the hyphen to the word
            masked_word = self._add_back_hyphen(masked_word)

            masked_word_list[self._word_position] = masked_word

            line = self._check_line_length(masked_word_list,
                                           self._word_position)

            if self._use_symspell:
                # when the spellchecker returns > 1 suggestions, use BERT
                # masking the miss-spelled word to predict possible words given
                # the context
                if len(suggestions) > 1:

                    if (self._word_position in self.apostrophe_words) and \
                            ('beginning' in self.apostrophe_words[
                                self._word_position]):
                        sug_terms = [sug.term for sug in suggestions[:10]]
                    else:
                        sug_terms = \
                            [str(" " + sug.term) for sug in suggestions[:10]]

                    # call BERT with targets
                    sug_terms_set = set(sug_terms)
                    try:
                        predict = self._fill_mask(line, targets=sug_terms)
                    except IndexError:
                        return

                    """
                    if ((len(self._line) > 6) and (self._word_position > 0)
                            and (self._word_position < len(self._line)-1)):
                        print(f"line: {self._line}\nword: {self._word}\n")
                        for prediction in predict:
                            print(f"prediction: {prediction['token_str']}\tscore: {prediction['score']}")
                    """

                    predict_sug = [prediction for prediction in predict if
                                   (prediction['token_str'] in sug_terms_set)
                                   and (prediction['score'] > 0.0001)]

                    if len(predict_sug) > 0:
                        self._word = predict_sug[0]['token_str'].strip()
                        return

            # Use BERT without targets to get top 30 predictions
            try:
                predict = self._fill_mask(line, top_k=30)
                predicted_words = \
                    [prediction["token_str"].strip() for prediction in predict]
            except IndexError:
                return

            if self._use_symspell:
                # check if any predicted word matches sym_spell output
                predict_sug = [word for word in predicted_words if word in [
                    sug.term.strip() for sug in suggestions]]
            else:
                # check that wue Levenshtein distance between the prediction and
                # the original word is less than 3 chars
                predict_sug = [word for word in predicted_words
                               if distance(word, self._word) < 3]

            if len(predict_sug) > 0:
                self._word = predict_sug[0]
                return

        if self._use_symspell:
            # check if the word is a composition of two words
            # don't consider Capitalized words in case they are names
            if not self._word[0].isupper():
                compound_word = self._sym_spell.lookup_compound(
                    self._word, max_edit_distance=0)
                word_compound = compound_word[0].term
                if word_compound != self._word:
                    self._word = word_compound.split()

        return

    def _check_line_length(self, masked_word_list: List[str], word_pos) -> str:
        """
        This method checks the token length of the line before being fed to
        BERT making sure is not larger than 514. If it is, then it returns a
        shorter version of the line. It uses recursivity to shorten the list
        until it has the desired length.
        Args:
            masked_word_list: list containing the words of the line and the
                masked word.
            word_pos: (int) position of the masked word in the list.

        Returns:
            line: string containing the words of the line and the masked word.
                The length of the tokenized words of the line is <= 514.
        """

        line = " ".join(masked_word_list)
        line = line.replace(" ,", ",").replace(" .", ".") \
            .replace(" :", ":").replace(" ;", ";").replace(" ?", "?") \
            .replace(" !", "!")
        model_inputs = self._fill_mask.tokenizer(
            line, return_tensors=self._fill_mask.framework)

        if len(model_inputs.input_ids[0]) <= 510:
            return line

        else:
            extra_rate_tokens = 510 / len(model_inputs.input_ids[0])
            new_length = int((len(masked_word_list) * extra_rate_tokens) - 1)

            if word_pos < new_length:  # word is in the beginning of the line
                masked_word_list = masked_word_list[:new_length]
                return self._check_line_length(masked_word_list, word_pos)

            elif len(line) - word_pos < new_length:  # word is in the end
                masked_word_list = masked_word_list[-new_length:]
                return self._check_line_length(
                    masked_word_list, word_pos - new_length)

            else:  # word is in the middle of the line
                masked_word_list = \
                    masked_word_list[word_pos - int(new_length / 2):
                                     word_pos + int(new_length / 2)]
                return self._check_line_length(
                    masked_word_list, int(new_length / 2))
