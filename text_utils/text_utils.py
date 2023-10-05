#!/usr/bin/env python3

"""
This module provides utility functions for working with text data.

It includes functions for calculating the length of tokenized text using the TikToken library,
calculating the embedding cost for a list of texts, and extracting file extensions from URLs.

Note: This module requires the 'tiktoken' library to be installed.
"""
from typing import List, TypeVar

T = TypeVar("T")


# create the length function
def tiktoken_len(text: str) -> int:
    """
    Returns the length of the tokenized version of the input text using the TikToken library.

    Args:
        text (str): The input text to tokenize.

    Returns:
        int: The length of the tokenized version of the input text.
    """
    import tiktoken

    tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


def embedding_cost(document: List[T]) -> float:
    """
    Calculates the embedding cost for a list of texts.

    Args:
    - document (List[T]): A list of texts to calculate the embedding cost for.

    Returns:
    - float: The embedding cost for the given list of texts.
    """

    total_tokens = sum([tiktoken_len(page.page_content) for page in document])
    print(f"Total tokens: {total_tokens}")
    return (total_tokens / 1000) * 0.0001


def return_url_extension(url: str) -> str:
    """
    Return the file extension from the given URL.

    Args:
    url (str): The URL to extract the file extension from.

    Returns:
    str: The file extension from the given URL.
    """
    from urllib.parse import urlparse
    import os

    parsed = urlparse(url)
    _, ext = os.path.splitext(parsed.path)
    return ext
