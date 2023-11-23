import os
from enum import Enum
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from src.data import transform, load
from src.exceptions import core as core_exceptions
from src.tools import custom_typing as ct, spacy_utils, utils as tools_utils, \
    transformers_utils
from src.tools.startup import settings, logger


class EmbeddingType(Enum):
    multilingual = 'multilingual'
    separate_languages = 'separate_languages'
    multilingual_transformer = 'multilingual_transformer'


class TfidfMethod(Enum):
    by_language = 'by_language'
    cross_language = 'cross_language'


def get_valid_texts_and_index_to_doc(
        df: pd.DataFrame, text_column: str) \
        -> Tuple[ct.StrList, Dict[int, str]]:
    """
    Given a pd.DataFrame and a text column, this function filters the empty
    strings and strings with only a space. Also, it creates a dictionary where
    the keys are the text position and the values are the documents id.

    Args:
        df (pd.DataFrame): a pd.DataFrame with text column.
        text_column (str): a column name to process.

    Returns:
        (Tuple[ct.StrList, Dict[int, str]]): a dict with the text position of
        the pd.DataFrame as a keys and the documents id as a values.
    """
    # Filter strings with only space or empty
    cond_is_space = df[text_column].str.isspace()
    cond_empty_str = df[text_column].str.strip().str.len() == 0
    cond_no_valid_str = cond_is_space | cond_empty_str
    # Get filtered texts for language
    lang_df = df.loc[~cond_no_valid_str, [text_column, 'doc_id']]
    texts = lang_df[text_column].values.tolist()
    doc_ids = lang_df['doc_id'].values.tolist()

    # Save the doc_id for each text
    index_to_doc = {
        idx: doc_id for idx, doc_id in zip(
            list(range(len(texts))), doc_ids)}

    return texts, index_to_doc


class Embedder:
    """
    This class takes care of computing embeddings for a corpus of
    text-only documents.

    Attributes:
        _corpus_df (pd.DataFrame): dataframe containing all documents we want
            to process. It must have columns "doc_id", "ca_text", "es_text",
            "it_text", "fra_text", "en_text". Column "doc_id" is given by
            <revista_id>_<publication_id>.

        _embedding_type (str): 'multilingual' or 'separate_languages'.

        _word_embeddings_dict (Dict[str, Dict[str, ct.Array]): nested dict
            where the first-level key is the  language and the second-level
            key is the  word. Contains the pre-trained word-embeddings.
            This field is filled automatically when embedding_type is
            'multilingual'.

        _weight_by_tfidf (boolean): if True, the word-vectors will be weighted
            by the relevance of that word within the document, computed
            through TF-IDF.

        _tfidf_method(str): there are two possible options:
            - 'cross_language': compute frequencies and relevance scores of
                words regardless of their language, putting them all together.
            - 'by_language': compute frequencies and relevance scores of words
                separately for each language.

        _tfidf_max_df (float): maximum document frequency parameter used by
            TF-IDF model (in percentage over the total).

        _tfidf_min_df (int): minimum document frequency parameter used by
            TF-IDF model (in units).

        _languages (ct.StrList): a list of langues codes.

        _embedding_size (int): the size of embedding vector to use.

        _embedding_path (str): the path to the embedding folder to store
            the embeddings.

        _word_relevances (Dict[str, Dict[str, Dict[str, float]]]): nested dict
            where the first-level key is the language, the second-level key is
            the document id, and the third-level key is the word; the value is
            the TF-IDF weight for that word in that document. This field is
            computed by method 'compute_word_relevances'.

        _doc_embeddings (Dict[str, ct.Array]): contains the embedding for each
            document in the corpus, indexed by document id. This field is
            computed by method 'compute_embeddings'.

        _spacy_models (Dict[str, str]): a dict with the language as keys and
            the model for these language as values.

    Methods:
        compute_word_relevances(): compute the TF-IDF weight for each word of
            each document in corpus.

        compute_embeddings(): compute embeddings for each document in the
            corpus.
    """

    def __init__(
            self, corpus_df: pd.DataFrame,
            embedding_type: Optional[str] = EmbeddingType.multilingual.value,
            weight_by_tfidf: Optional[bool] = True,
            tfidf_method: Optional[str] = TfidfMethod.cross_language.value,
            tfidf_max_df: Optional[float] = 0.8,
            tfidf_min_df: Optional[int] = 3,
            clean_texts: Optional[bool] = True,
            text_type: Optional[str] = '') -> None:
        """
        This is the constructor method of the Embedder class.It initializes the
        class attributes depends on the embedding type.

        Args:
            corpus_df (pd.DataFrame): a pd.DataFrame with the text corpus.
            embedding_type (Optional[str]): embedding approach to use to
                compute the word embeddings. There are three valid options
                "multilingual", "separate_language" or
                "multilingual_transformer". The default value
                is "multilingual".
            weight_by_tfidf (Optional[bool]): if it is True, the word-vectors
                will be weighted by the relevance of that word within the
                document, computed through TF-IDF. Otherwise, not. The default
                value is True.
            tfidf_method (Optional[str]): If 'weight_by_tfidf' is True, then
                a method to compute the TF-IDF is required. There are two
                available options:
                    - 'cross_language': compute frequencies and relevance
                    scores of words regardless of their language, putting
                    them all together.
                    - 'by_language': compute frequencies and relevance scores
                    of words separately for each language.
            tfidf_max_df (Optional[float]): maximum document frequency
                parameter used by TF-IDF model (in percentage over the total).
            tfidf_min_df (Optional[int]): minimum document frequency parameter
                used by TF-IDF model (in units).
            clean_texts (Optional, bool): if True, the texts of all the
                documents will be preprocessed with function 'clean_string'
                defined above. Keep in mind that functions
                'compute_word_relevances' and  'compute_doc_embeddings' assume
                that each text is a unique string without punctuation.
                The default value is True.
        """
        self._corpus_df: pd.DataFrame = corpus_df
        self._embedding_type: Optional[str] = embedding_type

        self._weight_by_tfidf: Optional[bool] = weight_by_tfidf
        self._tfidf_method: Optional[str] = tfidf_method
        self._tfidf_max_df: Optional[float] = tfidf_max_df
        self._tfidf_min_df: Optional[int] = tfidf_min_df

        self._word_relevances: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._doc_embeddings: Dict[str, ct.Array] = {}

        self._languages = None
        self._embedding_size = None
        self._embedding_path = settings['embeddings']['path']

        self._word_embeddings_dict: Dict[str, Dict[str, ct.Array]] = {}
        self._spacy_models: Dict[str, str] = {}

        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Device: {self._device}')

        # It refers if the text is corrected or not.
        self._text_type = text_type

        logger.info(f"Running embedding type '{embedding_type}'...")
        if embedding_type == EmbeddingType.multilingual.value:
            muse_settings = settings['embeddings']['muse']

            muse_vector_path = muse_settings['muse_vector_path']
            self._languages = muse_settings['languages']
            vectors_paths = \
                {la: f'{muse_vector_path}/{la}.txt' for la in self._languages}

            self._word_embeddings_dict = \
                {la: load.load_muse_embedding(vectors_paths[la])
                 for la in self._languages}
            self._embedding_size = muse_settings['embedding_size']

        elif self._embedding_type == EmbeddingType.separate_languages.value:
            spacy_settings = settings['embeddings']['spacy']

            self._spacy_models = spacy_settings['models']
            self._languages = list(self._spacy_models.keys())
            self._embedding_size = spacy_settings['embedding_size']

        elif embedding_type == EmbeddingType.multilingual_transformer.value:
            self._transformer_settings = \
                settings['embeddings']['multilingual_transformer']

        else:
            raise core_exceptions.EmbeddingTypeDoesNotExist(embedding_type)

        if clean_texts:
            self._clean_texts()

    @property
    def corpus_df(self) -> pd.DataFrame:
        """
        This is a getter method. It returns the "corpus_df" attribute.

        Returns:
            (pd.DataFrame): a pd.DataFrame with the corpus data.
        """
        return self._corpus_df

    @property
    def word_relevances(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        This is a getter method. It returns the "word_relevances" attribute.

        Returns:
            (Dict[str, Dict[str, Dict[str, float]]]): a dict with the computed
            word relevances.
        """
        return self._word_relevances

    @property
    def embeddings(self) -> Dict[str, ct.Array]:
        """
        This is a getter method. It returns the "embeddings" attribute.

        Returns:
            (Dict[str, ct.Array]): a dict with the "doc_id" as keys and the
            documents' embedding as values.
        """
        return self._doc_embeddings

    def _clean_texts(self) -> None:
        """
        Applies function 'clean_string' defined above to all texts
        in the corpus.
        """
        for lang in self._languages:
            self._corpus_df[f'{lang}_text'] = \
                self._corpus_df[f'{lang}_text'].apply(transform.clean_string)

    def _apply_tfidf(
            self, texts: ct.StrList, index_to_doc: Dict[int, str]) \
            -> Dict[str, Dict[str, float]]:
        """
        Applies TF-IDF to the list of texts received as parameters.

        Args:
            texts (ct.StrList): a list of texts on which to apply TF-IDF.
            index_to_doc (Dict[int, str]): mapping between position of the text
                in the list and the doc_id corresponding to that text.
                This is used to assign the TF-IDF scores to the
                right publication.

        Returns:
            (Dict[str, Dict[str, float]]): Nested dict where first-level keys
            are doc_ids, second-level keys are words and values are the
            TF-IDF score assigned to that word in that document.
        """
        logger.info(f'Running TF-IDF...')
        vectorizer = TfidfVectorizer(
            max_df=self._tfidf_max_df, min_df=self._tfidf_min_df)
        tfidf_matrix = vectorizer.fit_transform(texts).toarray()
        # Get the vocabulary
        vocabulary = vectorizer.get_feature_names_out()

        scores = {}
        for i, iter_matrix in enumerate(tfidf_matrix):
            # Get number of values greater than zeros
            values_gt_zero = iter_matrix[iter_matrix > 0].shape[0]
            # Sort by scores and filter only values greater than zero
            sorted_indices = np.argsort(iter_matrix, axis=None)[::-1]
            sorted_indices = sorted_indices[:values_gt_zero]
            # Add the score of words for each document
            scores[index_to_doc[i]] = {
                word: score for word, score in
                zip(vocabulary[sorted_indices],
                    iter_matrix[sorted_indices])}

        logger.info('Done.')
        return scores

    def compute_word_relevances(self) -> None:
        """
        Computes the relevance scores for each word in each document
        in the corpus, estimated through TF-IDF. Two methods can be
        applied, depending on 'tf_idf_method' class attribute:
            - 'cross_language': compute frequencies and
            relevance scores of words regardless of their
            language, putting them all together
            - 'by_language': compute frequencies and
            relevance scores of words separately for each language

        This method assumes that the texts in the corpus are already 
        clean (at least without punctuation). So, you should instantiate
        the class with clean_texts = True if your texts are un-processed
        texts.

        Fills the attribute word_relevances with a nested dict, where:
            - First level key is string 'cross_language' if
            tf_idf_method == 'cross_language', or language codes
            ('ca', 'es', etc.) if tf_idf_method == 'by_language'.
            - Second level keys are doc_ids.
            - Third-level keys are words.
            - Values are the TF-IDF score assigned to that word
            in that document.
        """
        if self._tfidf_method == TfidfMethod.cross_language.value:
            logger.info(f"Running cross_language")
            # Concatenate all texts
            texts = ""
            for lang in self._languages:
                lang_column = f'{lang}_text'
                texts = texts + " " + self._corpus_df[lang_column]

            # Create a column 'full_text' with the concatenation of all texts
            # of all languages in the document.
            full_text_column = 'full_text'
            self._corpus_df[full_text_column] = texts
            texts, index_to_doc = get_valid_texts_and_index_to_doc(
                self._corpus_df, full_text_column)
            # Compute TF-IDF.
            cross_lang = TfidfMethod.cross_language.value
            self._word_relevances[cross_lang] = self._apply_tfidf(
                texts, index_to_doc)

        elif self._tfidf_method == TfidfMethod.by_language.value:
            for lang in self._languages:
                logger.info(f"Running by_language for language '{lang}'")

                lang_column = f'{lang}_text'
                if lang_column in self._corpus_df:
                    texts, index_to_doc = get_valid_texts_and_index_to_doc(
                        self._corpus_df, lang_column)

                    # Compute TF-IDF.
                    self._word_relevances[lang] = self._apply_tfidf(
                        texts, index_to_doc)

            logger.info('Done.')

    def _get_words_relevances(
            self, lang: str, doc_id: str, split_texts: pd.Series) -> pd.Series:
        """
        This function creates a pd.Series with the relevance of each word
        in the document for the language.

        Args:
            lang (str): the language to get the word relevances.
            doc_id (str): the document id to get the word relevances.
            split_texts (pd.Series): a pd.Series with the words of the
                documents.

        Returns:
            (pd.Series): a pd.Series with the relevance for each word.
        """
        relevances_dict = self._word_relevances[lang].get(doc_id, {})
        return split_texts.map(relevances_dict.get).fillna(0)

    def _create_embeddings_filename(
            self, lang: str, doc_id: str, model_name: Optional[str] = '',
            text_type: Optional[str] = '') \
            -> str:
        """
        This function generates the output filename of the embedding.

        Args:
            lang (str): the language of the embedding to add in the embedding
                filename.
            doc_id (str): the document id of the embedding to add in the
                embedding filename.
            model_name (Optional[str]): the name of model name used. This is
                only for 'multilingual_transformer' type.

        Returns:
            (str): the embedding filename.
        """
        if text_type != '':
            text_type = f'_{text_type}'
        if self._embedding_type == EmbeddingType.separate_languages.value:
            filename = f'{self._spacy_models[lang]}_{doc_id}{text_type}.npy'
        elif self._embedding_type == \
                EmbeddingType.multilingual_transformer.value:
            filename = f'{model_name}_{doc_id}{text_type}.npy'
        else:
            raise ValueError('Error when trying to generate the '
                             'embedding filename.')
        return os.path.join(self._embedding_path, filename)

    def _compute_multilingual(self) -> None:
        """
        This method computes the "cross_language" approach. The original
        word-embedding are the pre-trained MUSE embeddings. This model aligns
        word-embeddings of different languages in the same vector space,
        so that words of different languages can be compared. The embedding of
        a document is computed as the weighted average of the MUSE embeddings
        of all words in the document, weighted by TF-IDF scores
        (if such option is selected).
        """
        with tqdm(self._corpus_df.iterrows(),
                  unit="it",
                  desc=f'Computing embeddings') as pbar:
            for _, row in pbar:
                embedding = np.zeros(self._embedding_size)
                normalizer = 0

                is_cross_method = \
                    self._tfidf_method == TfidfMethod.cross_language.value
                for lang in self._languages:
                    # Set the first-level key for word_relevances dict.
                    dict_key = TfidfMethod.cross_language.value \
                        if is_cross_method else lang

                    split_column = f'{lang}_split'

                    split_words = pd.Series(row[split_column], dtype='object')
                    # Get embeddings.
                    word_embeddings = split_words \
                        .map(self._word_embeddings_dict[lang])
                    # Get weights from precomputed word relevances dict if
                    # the word exists
                    # (see reference of compute_word_relevances).
                    if self._weight_by_tfidf:
                        words_weights = self._get_words_relevances(
                            dict_key, row['doc_id'], split_words)
                    else:
                        # Otherwise, the weights are always one.
                        words_weights = np.ones(split_words.shape)

                    # Multiply embeddings and weights.
                    lang_embedding = \
                        (word_embeddings * words_weights).sum()

                    # Update the embedding vector.
                    embedding += lang_embedding
                    # Update the normalizing factor.
                    normalizer += words_weights.sum()

                # Normalize the vector
                if normalizer > 0:
                    embedding /= normalizer

                # add the embedding to the dict.
                self._doc_embeddings[row['doc_id']] = embedding

    def _compute_separate_languages(self) -> None:
        """
        This method computes the "separate_language" approach. The base word
        embeddings constitute  pre-trained models tailored to individual
        languages. This approach entails the computation of the mean embedding
        representation for each document in the corpus. This mean
        representation is derived from the weighted aggregation of
        language-specific embeddings, wherein the weighting corresponds to the
        language's relevance within the document.
        """
        with tqdm(self._corpus_df.iterrows(),
                  unit="it",
                  desc=f'Computing embeddings') as pbar:
            for _, row in pbar:
                embedding_per_lang, tokens_per_lang = {}, {}
                for lang in self._languages:
                    # Check if there are words for this language
                    split_column = f'{lang}_split'
                    if row[split_column]:
                        # Count the number of words per language. This will
                        # be used as a weight when computing the final
                        # embedding for document.
                        tokens_per_lang[lang] = len(row[split_column])

                        # Compute embeddings.
                        output_filepath = self._create_embeddings_filename(
                            lang, row["doc_id"], text_type=self._text_type)
                        word_embeddings = spacy_utils.compute_word_embeddings(
                            row[split_column], self._spacy_models[lang],
                            output_filepath)

                        if self._weight_by_tfidf:
                            # Compute the average weight based on the relevance
                            # of each word in the document.
                            split_words = pd.Series(
                                row[split_column], dtype='object')
                            words_weights = self._get_words_relevances(
                                lang, row['doc_id'], split_words)

                            try:
                                lang_embedding = np.average(
                                    word_embeddings,
                                    weights=words_weights,
                                    axis=0)
                            except ZeroDivisionError as _:
                                # If a zero division error is raised, it means
                                # that all words have 0 relevance. The result
                                # should be an embedding of all zeros.
                                logger.warning(
                                    f"All the word weights for document "
                                    f"'{row['doc_id']}' are zero: "
                                    f"The embedding is setting to 0s.")
                                lang_embedding = np.zeros(self._embedding_size)
                        else:
                            lang_embedding = np.mean(word_embeddings, axis=0)

                        # Save the document language embedding.
                        embedding_per_lang[lang] = lang_embedding

                # Compute the average embedding per document using weights
                # based on the tokens for each language in the document.
                embeddings = pd.Series(embedding_per_lang)
                weights = pd.Series(tokens_per_lang)
                embedding = np.average(embeddings, weights=weights, axis=0)

                # Add the embedding to the dict.
                self._doc_embeddings[row['doc_id']] = embedding

    def _compute_multilingual_transformer(self) -> None:
        """
        This method employs the "multilingual_transformer" approach to
        calculate text embeddings. It processes the entire text to derive these
        embeddings, utilizing a multilingual transformer model for the task.
        Prior to embedding generation, the input text is partitioned into
        smaller chunks to facilitate processing. Subsequently, this approach
        computes the mean of all embeddings for each document, providing a
        summarized representation of the text's content.
        """
        # Generate a list of chunks for each content.
        logger.info(f'Generating chunks from text...')
        text_column = self._transformer_settings['text_column']
        self._corpus_df['text_chunks'] = self._corpus_df[text_column].apply(
            tools_utils.get_string_chunks,
            args=(self._transformer_settings['sequence_len'],)
        )
        logger.info('Done.')

        model_name = self._transformer_settings['model_name']
        transformer_params = {
            'device': self._device
        }
        # Create a Sentence Transformer instance.
        model = transformers_utils.create_sentence_transformer(
            model_name, **transformer_params)

        with tqdm(self._corpus_df.iterrows(),
                  unit="it",
                  desc=f'Computing embeddings') as pbar:
            for _, row in pbar:
                # Compute average embedding per document.
                output_filepath = self._create_embeddings_filename(
                    'multi', row["doc_id"], model_name, self._text_type)

                embedding = transformers_utils.generate_average_embedding(
                    model_name, row['text_chunks'], output_filepath,
                    model=model, **transformer_params)

                self._doc_embeddings[row['doc_id']] = embedding

    def compute_doc_embeddings(self) -> None:
        """
        Computes embeddings for all the documents in the corpus,
        with a prior step of word-relevance computation if
        'weight_by_tfidf' is True. Two methods are possible:
            - 'multilingual' (see the "multilingual" method).
            - 'separated_languages' (see the "_compute_separate_languages"
                method).

        This method, assumes that the texts in the df are clean
        (at least without punctuation).

        Fills the doc_embeddings attribute with a dict, where keys
        are doc_ids and values are the resulting document embeddings.
        """
        word_embeddings_types = [
            EmbeddingType.multilingual.value,
            EmbeddingType.separate_languages.value,
        ]
        if self._embedding_type in word_embeddings_types:
            if self._weight_by_tfidf and self._word_relevances == {}:
                self.compute_word_relevances()

            # Create a split column for each language.
            for lang in self._languages:
                lang_column = f'{lang}_text'
                if lang_column in self._corpus_df:
                    split_column = f'{lang}_split'
                    self._corpus_df[split_column] = \
                        self._corpus_df[lang_column].str.split()

            if self._embedding_type == EmbeddingType.multilingual.value:
                self._compute_multilingual()
            elif self._embedding_type == EmbeddingType.separate_languages.value:
                self._compute_separate_languages()

        elif self._embedding_type == \
                EmbeddingType.multilingual_transformer.value:
            return self._compute_multilingual_transformer()

        else:
            raise NotImplementedError(
                f"The embedding_type {self._embedding_type} is not supported")

        logger.info('Done')
