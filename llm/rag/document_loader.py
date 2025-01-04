from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.schema import Document

import uuid

def load_index(data_dir):
    documents = load_documents(data_dir)
    return VectorStoreIndex.from_documents(
    documents,
    show_progress=True
    )

def load_documents(data_dir):
    return SimpleDirectoryReader(data_dir, recursive=True, exclude = ['*.pptx', '*.xls', '*.xlsx', '*.pdf']).load_data()

def split_documents_by_line(documents):
    documents_result: list[Document] = []
    for doc in documents:
        for text in doc.text.split("\n"):
            if text.strip() == "":
                continue
            documents_result.append(Document(
                id_=str(uuid.uuid4()),
                text=text,
                metadata={
                    "file_path": doc.metadata["file_path"],
                    "file": doc.metadata["file_path"]
                    }
            ))
    return documents_result