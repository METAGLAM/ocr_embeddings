import warnings

from tqdm import tqdm
from transformers import logging as logging_t

from src.data.ocr_elements import OcrText, OcrLine, OcrWord
from src.data.ocr_management import OcrManagement
from src.data.resource_loader import ResourceLoader
from src.tools.startup import logger, settings

warnings.filterwarnings("ignore")

# Hide transformer's token warnings.
logging_t.get_logger().setLevel(logging_t.ERROR)


def __main__():
    ocr_settings = settings["ocr"]

    resources = ResourceLoader(ocr_settings["load"])
    logger.info('Creating the vocabularies...')
    resources.create_vocabularies()

    if ocr_settings["transform"]["steps"]["use_symspell"]:
        logger.info('Loading the spellcheckers...')
        resources.load_spellchecker()

    if ocr_settings["transform"]["steps"]["use_bert"]:
        logger.info('Loading BERT models...')
        resources.load_bert()

    logger.info('Looking for previous executions and creating folder to save '
                'corrected OCRs...')
    ocr_management = OcrManagement(ocr_settings)
    logger.info('Preparing list of docs to process...')
    docs_to_process = ocr_management.list_docs_to_process()

    logger.info('Starting the processing and correction of the documents...')
    # iterate through files
    for i in tqdm(range(len(docs_to_process)), desc='Progress'):
        text_processor = OcrText(ocr_settings, docs_to_process[i],
                                 resources.vocabularies)

        # text_lines = text_processor.extract_lines()  # lines in string format

        # paragraphs in string format
        text_paragraphs = text_processor.extract_paragraphs()

        revista, file = docs_to_process[i]
        file_words = {}
        processed_text = []

        # iterate through lines
        for j in tqdm(range(len(text_paragraphs)), desc=f'{revista} - {file}'):
            line = text_paragraphs[j]
            line_processor = OcrLine(line, resources.vocabularies, file_words)

            lang = line_processor.lang  # extract line lang
            line_words = line_processor.line_words  # list of words
            file_words = line_processor.file_words  # add lang line to the dict

            processed_line = []
            apostrophe_words = {}
            hyphen_words = {}

            misspelled_words = False

            # iterate through words
            for word_position, word in enumerate(line_words):
                word_processor = OcrWord(
                    word, line_words, lang, word_position, resources,
                    ocr_settings, apostrophe_words, hyphen_words, file_words)

                if ocr_settings["save"]["file"]["processed_text"]["save"] or \
                        ocr_settings["save"]["file"]["processed_lines"][
                            "save"]:
                    processed_word = word_processor.process_word()
                    processed_line.append(processed_word)
                else:
                    _ = word_processor.process_word()

                apostrophe_words = word_processor.apostrophe_words
                hyphen_words = word_processor.hyphen_words
                file_words = word_processor.file_words

                if word_processor.originally_misspellen:
                    misspelled_words = True

            if ocr_settings["save"]["file"]["processed_text"]["save"] or \
                    ocr_settings["save"]["file"]["processed_lines"]["save"]:
                # convert list of processed words into a line string
                processed_line = (" ".join(processed_line)) \
                    .replace(" ,", ",").replace(" .", ".").replace(" :", ":") \
                    .replace(" ;", ";").replace(" ?", "?").replace(" !", "!")
                processed_text.append(processed_line + '\n')

            if ocr_settings["save"]["file"]["processed_lines"]["save"]:
                ocr_management.write_processed_lines(
                    misspelled_words, line, processed_line, revista, file, j,
                    lang)

        if ocr_settings["save"]["file"]["processed_text"]["save"]:
            ocr_management.write_processed_ocr(revista, file, processed_text)
        if ocr_settings["save"]["file"]["all_words"]["save"]:
            ocr_management.update_words_json(file_words, revista, file)
        if ocr_settings["save"]["file"]["statistics"]["save"]:
            ocr_management.compute_statistics(file_words, revista, file)


if __name__ == "__main__":
    __main__()
