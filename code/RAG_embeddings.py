from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.globals import set_verbose
set_verbose(True)

loader = DirectoryLoader('/Users/ridhipurohit/Documents/Winter 2024/Generative AI/Assignments/Final/docs',\
                          glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
chunk_size_value = 5000
chunk_overlap=1000


def get_docembeddings():
    # split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size_value,\
                                                chunk_overlap=chunk_overlap,length_function=len)
    texts = text_splitter.split_documents(documents)
    
    # create doc embeddings
    print("Start doc embeddings generation...")
    docembeddings = FAISS.from_documents(texts, OpenAIEmbeddings())

    # saving embeddings to persist beyond the session
    docembeddings.save_local("statistics_llm_faiss_index")

    print("Embeddings generation completed.")


if __name__ == "__main__":
   
   # create and save embeddings for efficient retrieval
    get_docembeddings()
    