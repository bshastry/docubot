from text_utils.text_utils import (
    tiktoken_len,
    num_tokens_and_cost,
    return_url_extension,
)
import unittest

# As of 2021-10-20, the cost of embedding a single token using OpenAI is $0.0000001
EMBEDDING_COST_PER_TOKEN = 0.0000001


class TestTextUtils(unittest.TestCase):
    def test_tiktoken_len(self):
        self.assertEqual(tiktoken_len("Hello, world!"), 4)
        self.assertEqual(tiktoken_len("This is a sentence."), 5)
        self.assertEqual(tiktoken_len("This is a longer sentence with more words."), 9)

    def test_embedding_cost(self):
        class Page:
            def __init__(self, page_content):
                self.page_content = page_content

        document = [
            Page("This is the first page."),
            Page("This is the second page."),
            Page("This is the third page."),
        ]
        num_tokens, cost = num_tokens_and_cost(document)
        self.assertEqual(num_tokens, 18)
        self.assertAlmostEquals(cost, num_tokens * EMBEDDING_COST_PER_TOKEN)

        document = [
            Page("This is a short page."),
            Page("This is a longer page with more words."),
            Page("This is the longest page of them all, with many many words."),
        ]
        num_tokens, cost = num_tokens_and_cost(document)
        self.assertEqual(num_tokens, 29)
        self.assertAlmostEqual(cost, (num_tokens * EMBEDDING_COST_PER_TOKEN))

    def test_return_url_extension(self):
        self.assertEqual(
            return_url_extension("https://www.example.com/index.html"), ".html"
        )
        self.assertEqual(
            return_url_extension("https://www.example.com/path/to/file.txt"), ".txt"
        )
        self.assertEqual(
            return_url_extension("https://www.example.com/path/to/image.jpg"), ".jpg"
        )
        self.assertEqual(
            return_url_extension("https://www.example.com/path/to/doc.pdf"), ".pdf"
        )
        self.assertEqual(
            return_url_extension("https://www.example.com/path/to/README.md"), ".md"
        )
        self.assertEqual(
            return_url_extension("https://www.example.com/path/to/ms.docx"), ".docx"
        )


if __name__ == "__main__":
    unittest.main()
