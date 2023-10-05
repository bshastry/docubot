#!/usr/bin/env python3

"""
This module provides functions for loading various types of documents.

It includes functions for loading PDF documents, .docx documents, markdown documents,
plain text documents, documents from Wikipedia, and documents from files or URLs.

Note: This module requires the 'langchain' library to be installed.
"""

from typing import List, TypeVar

T = TypeVar("T")


def load_pdf_document(document_name: str, verbose: bool = False) -> List[T]:
    """
    Load a PDF document using PyPDFLoader.

    Args:
        document_name (str): The name of the PDF document to load.
        verbose (bool, optional): Whether to print a message indicating that the document is being loaded. Defaults to False.

    Returns:
        List[T]: A list of objects representing the loaded PDF document.
    """
    from langchain.document_loaders import PyPDFLoader

    if verbose:
        print(f"Loading {document_name}")
    return PyPDFLoader(document_name).load()


def load_docx_document(document_name: str, verbose: bool = False) -> List[T]:
    """
    Load a document in .docx format using the Docx2txtLoader class.

    Args:
        document_name (str): The name of the document to load.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        List[T]: A list of strings representing the document's contents.
    """
    from langchain.document_loaders import Docx2txtLoader

    if verbose:
        print(f"Loading {document_name}")
    return Docx2txtLoader(document_name).load()


def load_markdown_document(document_name: str, verbose: bool = False) -> List[T]:
    """
    Load a markdown document using the UnstructuredMarkdownLoader.

    Args:
        document_name (str): The name of the markdown document to load.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        List[T]: A list of objects loaded from the markdown document.
    """
    from langchain.document_loaders import UnstructuredMarkdownLoader

    if verbose:
        print(f"Loading {document_name}")
    return UnstructuredMarkdownLoader(document_name).load()


def load_txt_document(document_name: str, verbose: bool = False) -> List[T]:
    """
    Load a text document using the TextLoader class from langchain.document_loaders.

    Args:
        document_name (str): The name of the document to load.
        verbose (bool, optional): Whether to print a message indicating that the document is being loaded. Defaults to False.

    Returns:
        List[T]: A list of the loaded document's contents.
    """
    from langchain.document_loaders import TextLoader

    if verbose:
        print(f"Loading {document_name}")
    return TextLoader(document_name).load()


def load_from_wikipedia(
    query: str, lang: str = "en", load_max_docs: int = 2
) -> List[T]:
    """
    Load documents from Wikipedia based on the given query and language.

    Args:
        query (str): The query to search for on Wikipedia.
        lang (str, optional): The language of the Wikipedia to search. Defaults to "en".
        load_max_docs (int, optional): The maximum number of documents to load. Defaults to 2.

    Returns:
        List[T]: A list of documents loaded from Wikipedia.
    """
    from langchain.document_loaders import WikipediaLoader

    return WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs).load()


def load_document(document_name: str) -> List[T]:
    """
    Load a document from a file or URL.

    Args:
        document_name (str): The name of the document file or URL.

    Returns:
        List[T]: A list of the loaded document's contents.

    Raises:
        None.
    """
    import os
    import validators
    from text_utils.text_utils import return_url_extension

    if validators.url(document_name):
        if return_url_extension(document_name) == ".pdf":
            return load_pdf_document(document_name)
        else:
            print("Unsupported file extension")
            return None
    else:
        _, extension = os.path.splitext(document_name)
        if extension == ".pdf":
            return load_pdf_document(document_name)
        elif extension == ".docx":
            return load_docx_document(document_name)
        elif extension == ".md":
            return load_markdown_document(document_name)
        elif extension == ".txt":
            return load_txt_document(document_name)
        else:
            print("Unsupported file extension")
            return None


def chunk_data(
    data: List[T], chunk_size: int = 512, chunk_overlap: int = 20
) -> List[T]:
    """
    Splits the input data into chunks of specified size using a RecursiveCharacterTextSplitter.

    Args:
        data (str): The input data to be split into chunks.
        chunk_size (int, optional): The size of each chunk. Defaults to 512.

    Returns:
        List[T]: A list of chunks, where each chunk is of type T.
    """
    from text_utils.text_utils import tiktoken_len
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(data)
    return chunks
