from langchain.llms import HuggingFaceHub
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain, ConversationChain

import gradio as gr
from dotenv import load_dotenv
import os

default_persist_directory = "./db"

load_dotenv()


def split_doc(list_doc_obj):
    print("Entered split _doc method")
    pages = []

    loaders = [PyPDFLoader(x) for x in list_doc_obj]

    for loader in loaders:
        pages.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)

    text_chunks = text_splitter.split_documents(pages)
    print("Done with splitting the doc")

    return text_chunks

def initialize_db(list_doc_obj):
    print("Entering initialize_db method")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    documents = split_doc(list_doc_obj)

    vector_db = Chroma.from_documents(embedding=embeddings, documents=documents, persist_directory=default_persist_directory)

    retriever = vector_db.as_retriever()
    print("Initializing DB done")

    return retriever

def initialize_llm_chain(list_doc_obj):
    print("Entered initialize_llm_chain method")
    list_file_obj = [x.name for x in list_doc_obj]

    retriever = initialize_db(list_file_obj)

    HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

    llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                         huggingfacehub_api_token=HUGGINGFACE_API_KEY,
                         model_kwargs={'temperature':0.8, 'max_token':2048, 'top_k':3})
    
    memory = ConversationBufferMemory(output_key='chat_history', return_messages=True)
    
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever,
                                                     memory=memory, return_source_documents=True)
    
    print("Initializing llm done")

    return qa_chain

def format_chat_history(chat_history):
    print("Entered format_chat_history method")
    formatted_chat_history = []

    for user_message, bot_message in chat_history:
        formatted_chat_history.append(f"User : {user_message}")
        formatted_chat_history.append(f"Assistant : {bot_message}")

    print(formatted_chat_history)
    print("format chat history done")

    return formatted_chat_history

def initialize_conversation(qa_chain, message, chat_history):
    formatted_chat_history = format_chat_history(chat_history)

    print("Entered initializing conversation method")
    print(type(qa_chain))
    response = qa_chain({'question': message, 'chat_history': formatted_chat_history})

    response_answer = response['answer']

    response_source = response['source_documents']

    response_source_content1 = response_source[0].page_content.strip()

    response_source_content2 = response_source[2].page_content.strip()

    new_chat_history = chat_history + [message, response_answer]

    return response_answer, new_chat_history

def chat_GUI():
    with gr.Blocks(theme="soft") as chat_GUI:
        vector_db = gr.State()
        qa_chain = gr.State()

        gr.Markdown(
        """<center><h2>PDF-based chatbot (powered by LangChain and open-source LLMs)</center></h2>
        <h3>Ask any questions about your PDF documents, along with follow-ups</h3>
        <b>Note:</b> This AI assistant performs retrieval-augmented generation from your PDF documents. \
        When generating answers, it takes past questions into account (via conversational memory), and includes document references for clarity purposes.</i>
        <br><b>Warning:</b> This space uses the free CPU Basic hardware from Hugging Face. Some steps and LLM models used below (free inference endpoints) can take some time to generate an output.<br>
        """)
        with gr.Row():
            documents = gr.File(height=100, file_count='multiple', file_types=['pdf'], label="Place your PDF files. Single or multiple")
        with gr.Row():
            db_btn = gr.Button("Initialize Database")
        chatbot = gr.Chatbot(height=300)
        with gr.Row():
            msg = gr.Textbox(placeholder="Type your message here", container=True)
        with gr.Row():
            submmit_btn = gr.Button("Submit")
            clear_btn = gr.ClearButton([chatbot, msg])
        
        db_btn.click(initialize_llm_chain, 
                     inputs=[documents], outputs=[qa_chain])

        msg.submit(initialize_conversation,
                   inputs=[qa_chain, msg, chatbot], outputs=[msg, chatbot], queue=False)
        
        submmit_btn.click(initialize_conversation,
                          inputs=[qa_chain, msg, chatbot], outputs=[msg, chatbot], queue=False)
        
        clear_btn.click(lambda: [None], inputs=None, outputs=[chatbot], queue=False)

    chat_GUI.queue().launch(debug=True)

if __name__ == "__main__":
    chat_GUI()











