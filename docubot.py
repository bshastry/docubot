#!/usr/bin/env python3
from typing import List, TypeVar
from langchain.vectorstores import Pinecone

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
    root, ext = os.path.splitext(parsed.path)
    return ext


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

    if validators.url(document_name):
        if return_url_extension(document_name) == ".pdf":
            return load_pdf_document(document_name)
        else:
            print("Unsupported file extension")
            return None
    else:
        name, extension = os.path.splitext(document_name)
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


def chunk_data(data: str, chunk_size: int = 512) -> List[T]:
    """
    Splits the input data into chunks of specified size using a RecursiveCharacterTextSplitter.

    Args:
        data (str): The input data to be split into chunks.
        chunk_size (int, optional): The size of each chunk. Defaults to 512.

    Returns:
        List[T]: A list of chunks, where each chunk is of type T.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=20,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(data)
    return chunks


def create_index(index_name: str) -> None:
    """
    Creates a Pinecone index with the given name.

    Args:
        index_name (str): The name of the index to create.

    Raises:
        ValueError: If the index name is already taken.
        pinecone.exceptions.PineconeException: If there is an error creating the index.
    """
    import pinecone

    if index_name in pinecone.list_indexes():
        raise ValueError(f"Index {index_name} already exists.")
    try:
        print(f"Creating index {index_name}... ", end="")
        pinecone.create_index(name=index_name, metric="cosine", dimension=1536)
        print(f"Done")
    except pinecone.exceptions.PineconeException as e:
        raise pinecone.exceptions.PineconeException(f"Error creating index: {str(e)}")


def create_vector_store(index_name: str, chunks: List[T]) -> Pinecone:
    """
    Populates a Pinecone index with embeddings for the input documents (chunks).

    Args:
        index_name (str): The name of the index to populate.
        chunks (List[T]): A list of data chunks to index.

    Returns:
        Pinecone: A Pinecone vector store object containing the indexed data.

    Raises:
        ValueError: If the index name does not exist.
        pinecone.exceptions.PineconeException: If there is an error indexing the documents.
    """
    import pinecone
    from langchain.vectorstores import Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()
    if index_name not in pinecone.list_indexes():
        raise ValueError(f"Index {index_name} does not exist.")
    try:
        print(f"Indexing documents... ", end="")
        vector_store = Pinecone.from_documents(
            chunks, embeddings, index_name=index_name
        )
        print(f"Done")
    except pinecone.exceptions.PineconeException as e:
        raise pinecone.exceptions.PineconeException(
            f"Error indexing documents: {str(e)}"
        )
    return vector_store


def fetch_vector_store(index_name: str) -> Pinecone:
    """
    Retrieves embeddings from an already existing Pinecone index.

    Args:
        index_name (str): The name of the index to fetch embeddings from.

    Returns:
        Pinecone: A Pinecone vector store object containing the fetched embeddings.

    Raises:
        ValueError: If the index name does not exist.
        pinecone.exceptions.PineconeException: If there is an error fetching the embeddings.
    """
    import pinecone
    from langchain.vectorstores import Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()
    if index_name not in pinecone.list_indexes():
        raise ValueError(f"Index {index_name} does not exist.")
    try:
        vector_store = Pinecone.from_existing_index(
            index_name=index_name, embedding=embeddings
        )
    except pinecone.exceptions.PineconeException as e:
        raise pinecone.exceptions.PineconeException(
            f"Error fetching embeddings: {str(e)}"
        )
    return vector_store


def delete_pinecone_index(index_name: str) -> None:
    """
    Deletes a Pinecone index with the given name or all indices if index_name is "all".

    Args:
        index_name (str): Name of the index to delete. If "all", all indices will be deleted.

    Returns:
        None
    """
    import pinecone

    if index_name == "all":
        indices = pinecone.list_indexes()
        print(f"Deleting all {len(indices)} indices... ", end="")
        for index in indices:
            pinecone.delete_index(index)
        print("Done")
    else:
        print(f"Deleting index {index_name}... ", end="")
        pinecone.delete_index(index_name)
        print("Done")


def answer_question(question: str, vector_store, num_neighbors=5):
    """
    This function takes a question as input and returns an answer using a retrieval-based question answering system.

    Args:
    - question (str): The question to be answered.
    - vector_store: A vector store object used for retrieving similar documents.
    - num_neighbors (int): The number of similar documents to retrieve.

    Returns:
    - The answer to the input question.
    """
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import RetrievalQA

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": num_neighbors}
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )
    return chain.run(question)


def answer_question_session(
    question: str, vector_store, num_neighbors=5, chat_history=[]
):
    """
    This function takes a question, a vector store, and an optional number of neighbors and chat history.
    It uses the LangChain library to create a conversational retrieval chain, which is used to generate an answer to the question.
    The answer is then appended to the chat history.
    The function returns a dictionary containing the answer and the source documents used to generate the answer, as well as the updated chat history.

    Args:
        question (str): The question to be answered.
        vector_store: The vector store used to retrieve similar documents.
        num_neighbors (int, optional): The number of similar documents to retrieve. Defaults to 5.
        chat_history (list, optional): A list of tuples containing the question and answer pairs from previous conversations. Defaults to an empty list.

    Returns:
        tuple: A tuple containing a dictionary with the answer and source documents, and the updated chat history.
    """
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import ConversationalRetrievalChain

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": num_neighbors}
    )
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, return_source_documents=True
    )
    result = conversational_chain({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))
    return result, chat_history


def format_citations(source_docs: List[T]) -> str:
    """
    Formats a list of source documents into a string of citations.

    Args:
        source_docs (List[T]): A list of source documents.

    Returns:
        str: A string of formatted citations.
    """
    # Iterate through source_docs and reference a field in each index called metadata
    citation_list = []
    for index, doc in enumerate(source_docs):
        source_document_name = doc.metadata["source"]
        page = doc.metadata.get("page")
        page_str = str(int(page)) if page else "not available"
        citation_list.append(
            "{}: {}, page {}".format(index, source_document_name, page_str)
        )
    return "\n".join(citation_list)


def chatbot(vector_store):
    """
    A chatbot function that allows users to ask questions and receive answers with citations.

    Args:
        vector_store (list): A list of vectors used for semantic search.

    Returns:
        None
    """
    import time

    i = 1
    print("Welcome to the chatbot. Type 'quit' or 'exit' to exit")
    chat_history = []
    while True:
        question = input(f"Question #{i}: ")
        if question.lower() in ["quit", "exit"]:
            print("Exiting chatbot")
            time.sleep(2)
            break
        result, chat_history = answer_question_session(
            question, vector_store=vector_store, chat_history=chat_history
        )
        print(f"\nAnswer #{i}: {result['answer']}")
        # Join vector in result['source_document'] into a string seperated by newline character
        citations = format_citations(result["source_documents"])
        print(f"\nCitations #{i}:\n{citations}")
        print(f"\n {'-' * 50} \n")
        i += 1


def load_document_and_chunk(document_name: str, chunk_size=512):
    """
    Load a document and chunk it into smaller pieces.

    Args:
        document_name (str): The name of the document to load.
        chunk_size (int): The size of each chunk in bytes. Default is 512.

    Returns:
        list: A list of chunks, where each chunk is a string.
    """
    data = load_document(document_name)
    chunks = chunk_data(data, chunk_size=chunk_size)
    return chunks


# Build knowledge base
def build_kb(data_directory: str) -> List[T]:
    """
    Builds a knowledge base by loading documents from the given data directory.

    Args:
        data_directory (str): The path to the directory containing the documents.

    Returns:
        List[T]: A list of chunks, where each chunk contains 512 tokens.
    """
    import os
    from tqdm import tqdm

    list_of_doc_names = []
    # List of supported document types
    extensions = (".pdf", ".docx", ".md", ".txt")
    for root, _, files in os.walk(data_directory):
        for file in files:
            if file.endswith(extensions):
                list_of_doc_names.append(os.path.join(root, file))
            else:
                continue

    # This is the knowledge base that is chunked into 512 tokens per chunk
    data = []
    print(f"Splitting documents into chunks")
    for doc_name in tqdm(list_of_doc_names):
        data.extend(load_document(doc_name))

    print(f"There are {len(data)} pages in the knowledge base")
    chunks = chunk_data(data, chunk_size=512)
    # Debugging
    print(f"Cost of embedding is: ${embedding_cost(chunks):.4f}")
    return chunks


def init():
    from dotenv import load_dotenv, find_dotenv
    import pinecone
    import os

    # Load API keys
    load_dotenv(find_dotenv(), override=True)
    pinecone.init(
        api_key=os.environ.get("PINECONE_API_KEY"),
        environment=os.environ.get("PINECONE_ENV"),
    )


def docubot():
    """
    This function is the entry point of the DocuBot application. It creates an argparse to parse data directory which is a mandatory argument.
    It loads API keys and creates or loads existing index and instantiate QA chatbot.
    """
    import argparse
    import pinecone

    parser = argparse.ArgumentParser(description="DocuBot")
    parser.add_argument("docs_dir", type=str, help="Path to the documents directory")
    parser.add_argument(
        "--index_name",
        type=str,
        default="docubot-index",
        help="Descriptive name for backend vector db index",
    )
    parser.add_argument(
        "--reload",
        type=str,
        help="Name of index to reload. If 'all', all indices will be deleted.",
    )
    args = parser.parse_args()
    docs_directory = args.docs_dir
    index_name = args.index_name
    reload_index_name = args.reload
    print(f"Instantiating DocuBot for {docs_directory}")
    init()

    vector_store: Pinecone = None
    if reload_index_name:
        # Delete all pinecone indices. As a free user, we can only have one index at a time.
        delete_pinecone_index(reload_index_name)
        create_index(index_name)
        vector_store = create_vector_store(index_name, chunks=build_kb(docs_directory))
    elif index_name in pinecone.list_indexes():
        vector_store = fetch_vector_store(index_name)
    else:
        create_index(index_name)
        vector_store = create_vector_store(index_name, chunks=build_kb(docs_directory))
    chatbot(vector_store=vector_store)


if __name__ == "__main__":
    docubot()
