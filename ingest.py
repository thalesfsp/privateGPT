import os
import glob
from typing import List
from dotenv import load_dotenv

from langchain.document_loaders import TextLoader, TomlLoader, JSONLoader, PDFMinerLoader, CSVLoader, UnstructuredMarkdownLoader
from custom_loaders.configparser import ConfigLoader
from custom_loaders.yaml import YAMLLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS


load_dotenv()


def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf8")
    elif file_path.endswith(".pdf"):
        loader = PDFMinerLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)
    elif file_path.endswith(".go") or file_path.endswith(".js") or file_path.endswith(".ts") or file_path.endswith(".py"):
        loader = TextLoader(file_path, encoding="utf8")
    elif file_path.endswith(".json"):
        loader = TextLoader(file_path, encoding="utf8")
    elif file_path.endswith(".toml"):
        loader = TomlLoader(file_path)
    elif file_path.endswith(".ini") or file_path.endswith(".cfg") or file_path.endswith(".env"):
        loader = ConfigLoader(file_path)
    elif file_path.endswith(".yaml"):
        loader = YAMLLoader(file_path)
    elif file_path.endswith(".md"):
        loader = UnstructuredMarkdownLoader(file_path)
    return loader.load()[0]


def load_documents(source_dir: str) -> List[Document]:
    # Loads all documents from source documents directory
    txt_files = glob.glob(os.path.join(source_dir, "**/*.txt"), recursive=True)
    pdf_files = glob.glob(os.path.join(source_dir, "**/*.pdf"), recursive=True)
    csv_files = glob.glob(os.path.join(source_dir, "**/*.csv"), recursive=True)
    code_files = glob.glob(os.path.join(source_dir, "**/*.go"), recursive=True) + \
                 glob.glob(os.path.join(source_dir, "**/*.js"), recursive=True) + \
                 glob.glob(os.path.join(source_dir, "**/*.ts"), recursive=True) + \
                 glob.glob(os.path.join(source_dir, "**/*.py"), recursive=True)
    json_files = glob.glob(os.path.join(source_dir, "**/*.json"), recursive=True)
    toml_files = glob.glob(os.path.join(source_dir, "**/*.toml"), recursive=True)
    config_files = glob.glob(os.path.join(source_dir, "**/*.ini"), recursive=True) + \
                   glob.glob(os.path.join(source_dir, "**/*.cfg"), recursive=True) + \
                   glob.glob(os.path.join(source_dir, "**/*.env"), recursive=True)
    yaml_files = glob.glob(os.path.join(source_dir, "**/*.yaml"), recursive=True)
    md_files = glob.glob(os.path.join(source_dir, "**/*.md"), recursive=True)
    
    all_files = txt_files + pdf_files + csv_files + code_files + json_files + toml_files + config_files + yaml_files + md_files
    return [load_single_document(file_path) for file_path in all_files]



def main():
    # Load environment variables
    persist_directory = os.environ.get('PERSIST_DIRECTORY')
    source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
    llama_embeddings_model = os.environ.get('LLAMA_EMBEDDINGS_MODEL')
    model_n_ctx = os.environ.get('MODEL_N_CTX')

    # Load documents and split in chunks
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"Loaded {len(documents)} documents from {source_directory}")
    print(f"Split into {len(texts)} chunks of text (max. 500 tokens each)")

    # Create embeddings
    llama = LlamaCppEmbeddings(model_path=llama_embeddings_model, n_ctx=model_n_ctx)
    
    # Create and store locally vectorstore
    db = Chroma.from_documents(texts, llama, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None


if __name__ == "__main__":
    main()
