from pymongo import MongoClient
from pymongo.server_api import ServerApi
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch

from  langchain_community.document_loaders.directory import DirectoryLoader,TextLoader
from langchain_community.document_loaders import DirectoryLoader

from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import gradio as gr
from gradio.themes.base import Base
import key_param



uri = "mongodb+srv://arpan12pallar:HQbDfpKAmGA9NLds@cluster0.ee2pwlc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# Create a new client and connect to the server
client = MongoClient(uri)
dbName = "taylorSwift_QA"
collectionName = "collection_of_text_blobs"
collection = client[dbName][collectionName]
embeddings = OpenAIEmbeddings(openai_api_key=key_param.openai_api_key)
#vectorStore = MongoDBAtlasVectorSearch(collection,embeddings)
vectorStore = MongoDBAtlasVectorSearch.from_connection_string(
        uri,
        f"{dbName}.{collectionName}",
        embeddings,
        index_name="vector_index"
    )
def query_data(query):
    
    docs = vectorStore.similarity_search(query,k=1)
    print(len(docs))
    as_output = docs[0].page_content

    llm = OpenAI(openai_api_key=key_param.openai_api_key,temperature=0)
    retriever = vectorStore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm,chain_type="stuff",retriever=retriever)
    retriever_output = qa.run(query)

    return as_output,retriever_output




try:
    db = client.taylorSwift_QA
    collection = db.collection_of_text_blobs 
    # for doc in collection.find():
    #     print(doc)
    #     doc['plot_embedding_hf'] = generate_embedding(doc['plot'])
    #     collection.replace_one({'_id':doc['_id']},doc)

    with gr.Blocks(theme=Base(),title="Question Answering App") as demo:
        gr.Markdown(
            """
                #Question Answering App using Atlas Vector Search
            """
            )
        textbox = gr.Textbox(label="Enter your Question")

        with gr.Row():
            button = gr.Button("Submit",variant="primary")
        with gr.Column():
            output1 = gr.Textbox(lines=1,max_lines=10,label="output with atlas")
            output2 = gr.Textbox(lines=1,max_lines=10,label="output with atlas+LLM")
        
        button.click(query_data,textbox,outputs=[output1,output2])

    demo.launch()
    
    
    
    #query_data("hello")


except Exception as e:
    print(e)
