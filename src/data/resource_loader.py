import os
from typing import Dict

import spacy
import torch
# library to spellcheck words statistically
from symspellpy import SymSpell

from src.tools import transformers_utils


class ResourceLoader:
    """
    This class takes care of loading the resources needed to process the
    text.

    Attributes:
        _load_settings: settings that define the languages, dictionary paths,
            and other variables needed to create the vocabularies, SymSpell
            instances, and BERT model that will be used in each language.

        vocabularies: dictionary with a key for every language containing a
            set with all the words of the vocabulary and a set with all the
            stop words.

        sym_spell: dictionary containing a SymSpell instance for each language.

        fill_mask: dictionary containing the fill_mask pipeline for each
            language.

    Methods:
        create_vocabularies(): create a dictionary containing every word of
            the vocabulary of a language and its stop words.

        load_spellchecker(): create a dictionary containing a SymSpell
            instance for each language.

        load_bert(): create a dictionary containing the fill_mask pipeline for
            each language.
    """

    def __init__(self, load_settings: Dict):
        """
        This is the constructor method of the ResourceLoader class. It
        initializes the class attributes depending on the load settings.

        Args:
            load_settings: settings that define the languages, languages
                dictionary paths, and other variables needed to create the
                vocabularies, SymSpell instances, and BERT model that will be
                used in each language.
        """
        self._load_settings = load_settings
        self.vocabularies = None
        self.sym_spell = None
        self.fill_mask = None
        self._device = "cuda:0" if (torch.cuda.is_available() and
                                    load_settings["use_cuda_if_available"]) \
            else "cpu"

    def create_vocabularies(self) -> None:
        """
        This method creates a vocabulary dictionary with a key for every
        language containing a set with all the words of the vocabulary and a
        set with all the stop words.

        Example:
        {
            "ca": {
                "vocabulary": {
                    'escola', 'única', 'interior', 'pell', 'minuts', 'set',
                    'entendre', 'posat', 'sistema', 'fort', ... },
                "stop_words": {
                    'vostre', 'quins', 'pel', 'seus', 'us', 'ells', 'la',
                    'estan', 'sols', 'esteu', ... }
            }
        }
        """
        # read dictionaries
        dict_path = self._load_settings["dictionary_path"]
        vocabularies = {}
        for language in self._load_settings["languages"]:
            lang_path = os.path.join(dict_path, language["lang"])
            lang_dict = open(lang_path).readlines()

            words_set = set(map(self._process_words, lang_dict))

            words_set = set(filter(
                lambda x: self._check_valid_words(x, language), words_set))

            # load initials set
            initials = \
                set(open(os.path.join(dict_path, 'initials')).readlines())

            words_set = words_set.union(initials)

            # Stopwords
            nlp = spacy.load(language["vocabularies"]["spacy_pipeline"])
            stop_words = nlp.Defaults.stop_words

            vocabularies[language["lang_code"]] = {
                "vocabulary": words_set,
                "stop_words": stop_words
            }

        # create non_identified key joining all words and stopword from every
        # language
        vocabs = [e[1]["vocabulary"] for e in vocabularies.items()]
        stop_words = [e[1]["stop_words"] for e in vocabularies.items()]
        vocabulary_all = set().union(*vocabs)
        stop_words_all = set().union(*stop_words)
        vocabularies["non_identified"] = {
            "vocabulary": vocabulary_all,
            "stop_words": stop_words_all
        }

        self.vocabularies = vocabularies

    @staticmethod
    def _process_words(word: str) -> str:
        """
        Remove break line characters.
        Args:
            word: word of the dictionary.

        Returns:
            input word without the break line char.
        """
        return word.replace("\n", "")

    @staticmethod
    def _check_valid_words(word: str, language: dict) -> bool:
        """
        Given a word and its language, check if the word is more than one
        character long and if it is two characters long, make sure it is not
        made of two consonants.
        Args:
            word: word to check whether is valid
            language: language of the word

        Returns:
            boolean stating whether the word is valid or not.
        """
        # remove one character words and two characters words when both are
        # consonants
        if len(word) == 1:
            return False
        elif ((len(word) == 2) and
              (word[0] not in language["vocabularies"]["vowels"]) and
              (word[1] not in language["vocabularies"]["vowels"])):
            return False
        else:
            return True

    def load_spellchecker(self) -> None:
        """
        This method creates a dictionary containing a SymSpell instance for each
        language.
        """
        dict_path = self._load_settings["dictionary_path"]
        sym_spell_dict = {}
        for language in self._load_settings["languages"]:
            language["spellchecker"]["corpus"] = \
                os.path.join(dict_path, language["spellchecker"]["corpus"])
            sym_spell = SymSpell(max_dictionary_edit_distance=2,
                                 prefix_length=7)
            status = sym_spell.load_dictionary(**language["spellchecker"])
            self._check_spellchecker(status, sym_spell, language["lang"])

            sym_spell_dict[language["lang_code"]] = sym_spell

        self.sym_spell = sym_spell_dict

    @staticmethod
    def _check_spellchecker(status: bool, sym_spell: SymSpell, lang: str) -> \
            None:
        """
        Check if the syn_spell instance of SymSpell has correctly loaded the
        dictionary. Raise an error otherwise.
        Args:
            status: output resulting from the load of the sym_spell dictionary.
            sym_spell: SymSpell instance to check.
            lang: language of the dictionary.

        Returns:

        """
        assert status, f"{lang} dictionary was not loaded successfully"

        assert ((len(sym_spell.words) > 1000) and
                (type(list(sym_spell.words.keys())[0]) == str) and
                (list(sym_spell.words.keys())[0].isalpha()) and
                (type(list(sym_spell.words.values())[0]) == int)), \
            f"{lang} dictionary was not loaded successfully"

    def load_bert(self) -> None:
        """
        This method creates a dictionary containing the fill_mask pipeline for
        each language.
        """
        fill_mask_dict = {}
        for language in self._load_settings["languages"]:
            # Get the params to initialize a 'fill-mask' pipeline.
            pipeline_params = {
                'task': language['pipeline_task'],
                'top_k': language['top_k'],
                'model': language["bert"],
                'tokenizer': language["bert"],
                'device': self._device
            }
            fill_mask = transformers_utils.generate_pipeline(pipeline_params)

            fill_mask_dict[language["lang_code"]] = fill_mask

        self.fill_mask = fill_mask_dict
