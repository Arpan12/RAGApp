from pymongo import MongoClient
from pymongo.server_api import ServerApi
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from  langchain_community.document_loaders.directory import DirectoryLoader,TextLoader
from langchain_community.document_loaders.pdf import UnstructuredPDFLoader,OnlinePDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
import gradio as gr
from gradio.themes.base import Base
import key_param


FILETYPE = ".pdf"

# Create a new client and connect to the server
client = MongoClient(key_param.MONGO_URI, server_api=ServerApi('1'))


dbName = "taylorSwift_QA"
collectionName = "collection_of_text_blobs"
collection = client[dbName][collectionName]
if(FILETYPE == ".txt"):
    loader = DirectoryLoader(r'sample_files', glob="**/*.txt",loader_cls=TextLoader)
    # Load documents from the specified directory
    doc = loader.load()
    print(type(doc))
elif(FILETYPE == ".pdf"):

    loader = UnstructuredPDFLoader("/home/arpan/Projects/AI apps/RAG_QA_Apps/sample_files/C++ Concurrency in Action Practical Multithreading - 1st Edition.pdf")

    doc = loader.load()



# Process the loaded documents
text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
texts = text_splitter.split_documents(doc)
print(type(texts))
# Load documents from the specified directory

embeddings = OpenAIEmbeddings(openai_api_key=key_param.openai_api_key)
vectorStore = MongoDBAtlasVectorSearch.from_documents(texts,embeddings,collection=collection)

