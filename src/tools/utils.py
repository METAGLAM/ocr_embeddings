from typing import List


def get_string_chunks(string: str, length: int) -> List[str]:
    """
    Given a string and a length, it splits the string into chunks of length.

    Args:
        string (str): The string to split.
        length (int): The length of each chunk.

    Returns:
        (List[str]): A list of string chunks.
    """
    return list(
        (string[0 + i:length + i] for i in range(0, len(string), length)))
