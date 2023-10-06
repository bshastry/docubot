# DocuBot

[![Run Tests](https://github.com/bshastry/docubot/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/bshastry/docubot/actions/workflows/tests.yml)![Bandit Workflow](https://img.shields.io/badge/Bandit-No%20Issues%20Found-brightgreen)![Coverage](https://img.shields.io/badge/Coverage-90%25-brightgreen.svg)

DocuBot is a command-line chatbot that answers questions using a knowledge base of documents provided by you.
It allows you to interactively get answers to questions with citations from the documents provided.
It is written in Python3.

## Supported Document Types

DocuBot supports the following document types:

- .pdf: Portable Document Format
- .docx: Microsoft Word Document
- .md: Markdown Document
- .txt: Plain Text Document

## Features

- Session based: DocuBot remembers previous interactions within the current session.
- Citations provided: DocuBot generates answers based on information from specific documents. It provides citations to these documents, including page numbers if available.
- Answer limitations: In some cases, DocuBot may not be able to provide an appropriate answer based on the given context. In such cases, it will indicate that it cannot answer the question accurately.

Please note that while DocuBot to provide accurate information with proper citations, there might be situations where it may not have access to the required resources or may not be able to generate an answer.

## Prerequisites

Before using DocuBot, you need to have the following:

- [OpenAI API key and linked credit card or a paid account](https://platform.openai.com/signup)
- [Pinecone API and ENVIRONMENT keys](https://www.pinecone.io/)

To avoid OpenAI rate-limiting issues, it is recommended to preload funds into your OpenAI account. This ensures that you have sufficient credits to make multiple requests during the document indexing phase.

**Note:** DocuBot provides an estimated cost of indexing documents at the beginning of the process. This helps you understand the potential cost implications before proceeding. Please review the estimated cost and ensure that you have sufficient funds in your OpenAI account to cover the indexing process.


## Installation

To use DocuBot, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/bshastry/docubot.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables:

   - In the `docubot` top-level directory, copy the `.env.template` file to a new file called `.env`.
   - Open the `.env` file and add the following API keys:
     - `PINECONE_API_KEY`: Your Pinecone API key
     - `PINECONE_ENV`: The environment where your Pinecone index is hosted (e.g., "gcp-starter")
     - `OPENAI_API_KEY`: Your OpenAI API key

   Make sure to replace the placeholder values with your actual API keys and ENV variables.


4. Collect documents you want DocuBot to work with in a local sub-directory:

   ```bash
   cd docubot && mkdir -p documents && cd documents
   <download_or_copy_documents_here>
   ```

   Since this project was internally developed at Ethereum, there is a scripts folder in which you can find an ethereum specific document downloader. To use the ethereum downloader script, do:

   ```bash
   cd docubot && ./scripts/bash/download_ethereum.sh ethereum-docs
   ```

   You could create a similar script for your specific use-case.

5. Run the `docubot.py` script:

   ```bash
   python3 docubot.py /path/to/documents/directory
   ```

   Please replace `/path/to/documents/directory` with the path to the directory that holds documents you want DocuBot to interface with (e.g., `ethereum-docs` from the previous step)

## Usage

Once DocuBot is running, you can start asking questions. Simply type your question and press Enter. To quit DocuBot, type "quit" or "exit".

## Examples

Here are some examples of questions you can ask DocuBot:

1. "What is Ethereum?"
2. "How does Ethereum work?"
3. "What is a smart contract?"
4. "Who created Ethereum?"
5. "What is the current price of Ether?"

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Acknowledgements

This project uses the following libraries:

- dotenv
- langchain
- pinecone
- tiktoken
- validators
- urllib

I would like to thank the protocol security team at the Ethereum foundation for giving me the freedom to pursue this project.
I would also like to thank Andrei Dumitrescu for a wonderful course on LangChain and OpenAI at Udemy that made prototyping DocuBot a breeze.

## Contributing

If you'd like to contribute to this project, please open an issue or submit a pull request. We welcome any feedback or improvements.

## Liability Information

DocuBot is released under the MIT license. Please note that while DocuBot is designed to provide useful information, it should not be considered a substitute for professional advice. The developers and contributors of DocuBot shall not be held liable for any damages or losses arising from the use of this application.

It is recommended to use DocuBot responsibly and exercise caution when relying on its responses. If in doubt, it is always a good idea to consult with domain experts or refer to trusted sources for accurate information.
