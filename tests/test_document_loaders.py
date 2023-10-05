#!/usr/bin/env python3

"""
This module contains unit tests for the document loaders module.

The document loaders module provides functions for loading various types of documents, such as PDF, DOCX, Markdown, and plain text files. It also includes a function for loading content from Wikipedia.

The unit tests in this module ensure that the document loading functions return the expected results. They also test the chunk_data function, which is used to split document content into smaller chunks.

The module includes the following test cases:

- test_load_pdf_document: Tests the load_pdf_document function by loading a PDF file and checking that the returned content is a list with a length greater than 0.

- test_load_docx_document: Tests the load_docx_document function by loading a DOCX file and checking that the returned content is a list with a length greater than 0.

- test_load_markdown_document: Tests the load_markdown_document function by loading a Markdown file and checking that the returned content is a list with a length greater than 0.

- test_load_txt_document: Tests the load_txt_document function by loading a plain text file and checking that the returned content is a list with a length greater than 0.

- test_load_from_wikipedia: Tests the load_from_wikipedia function by loading content from Wikipedia for a given query and checking that the returned content is a list with a length greater than 0.

- test_load_document: Tests the load_document function by loading various types of documents (PDF, DOCX, Markdown, and plain text files) and checking that the returned content is a list with a length greater than 0.

- test_chunk_data: Tests the chunk_data function by loading a plain text file, splitting its content into smaller chunks, and checking that each chunk has a length less than or equal to the specified chunk size.

To run the unit tests, execute this module as a script.
"""

import unittest
from document_loaders.document_loaders import (
    load_pdf_document,
    load_docx_document,
    load_markdown_document,
    load_txt_document,
    load_from_wikipedia,
    load_document,
    chunk_data,
)
from text_utils.text_utils import tiktoken_len


class TestDocumentLoaders(unittest.TestCase):
    def test_load_pdf_document(self):
        pdf_file = "test_files/test.pdf"
        pdf_contents = load_pdf_document(pdf_file)
        self.assertIsInstance(pdf_contents, list)
        self.assertGreater(len(pdf_contents), 0)

    def test_load_docx_document(self):
        docx_file = "test_files/test.docx"
        docx_contents = load_docx_document(docx_file)
        self.assertIsInstance(docx_contents, list)
        self.assertGreater(len(docx_contents), 0)

    def test_load_markdown_document(self):
        md_file = "test_files/test.md"
        md_contents = load_markdown_document(md_file)
        self.assertIsInstance(md_contents, list)
        self.assertGreater(len(md_contents), 0)

    def test_load_txt_document(self):
        txt_file = "test_files/test.txt"
        txt_contents = load_txt_document(txt_file)
        self.assertIsInstance(txt_contents, list)
        self.assertGreater(len(txt_contents), 0)

    def test_load_from_wikipedia(self):
        query = "Python programming language"
        wikipedia_contents = load_from_wikipedia(query)
        self.assertIsInstance(wikipedia_contents, list)
        self.assertGreater(len(wikipedia_contents), 0)

    def test_load_document(self):
        pdf_file = "test_files/test.pdf"
        docx_file = "test_files/test.docx"
        md_file = "test_files/test.md"
        txt_file = "test_files/test.txt"
        url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"

        pdf_contents = load_document(pdf_file)
        self.assertIsInstance(pdf_contents, list)
        self.assertGreater(len(pdf_contents), 0)

        docx_contents = load_document(docx_file)
        self.assertIsInstance(docx_contents, list)
        self.assertGreater(len(docx_contents), 0)

        md_contents = load_document(md_file)
        self.assertIsInstance(md_contents, list)
        self.assertGreater(len(md_contents), 0)

        txt_contents = load_document(txt_file)
        self.assertIsInstance(txt_contents, list)
        self.assertGreater(len(txt_contents), 0)

        url_contents = load_document(url)
        self.assertIsInstance(url_contents, list)

    def test_chunk_data(self):
        txt_file = "test_files/test.txt"
        txt_contents = load_document(txt_file)
        # Chunk size in tokens (not characters)
        chunk_size = 10
        # Number of tokens to overlap between chunks
        chunk_overlap = 5
        chunks = chunk_data(
            txt_contents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(tiktoken_len(chunk.page_content), chunk_size)


if __name__ == "__main__":
    unittest.main()
