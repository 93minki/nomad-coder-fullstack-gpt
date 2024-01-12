import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import (
    OpenAIEmbeddings,
    CacheBackedEmbeddings,
    OllamaEmbeddings,
)
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.callbacks.base import BaseCallbackHandler

st.set_page_config(
    page_title="Private",
    page_icon="💼",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()  # 나중에 무언가를 담을 수 있는 빈 위젯

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOllama(
    model="mistral:latest",
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)


# embed_file 함수의 입력(file)이 변경되지 않는한 실행되지 않는다.
@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/private_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/private_embeddings/{file.name}")
    spliter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=spliter)
    embeddings = OllamaEmbeddings(model="mistral:latest")
    cache_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cache_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        # 메세지를 화면에 표시함 역할에 따라 다르게
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """
#             Answer the question using ONLY the following context. If you don't know the answer
#             just say you don't know. DON't make anything up.

#             Context: {context}
#             """,
#         ),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("human", "{question}"),
#     ]
# )
prompt = ChatPromptTemplate.from_template(
    """
        Answer in korean the question using ONLY the following context. and do not use your training data. If you don't know the answer
        just say you don't know. DON't make anything up.
        Context: {context}
        Question:{question}
    """
)

st.title("Private GPT")

st.markdown(
    """
            Welcome!
            
            Use this chatbot to ask questions to an AI about your files!
            
            Upload your files on the sidebar.
            """
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx or .png file",
        type=["pdf", "txt", "docx", "png"],
    )

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            result = chain.invoke(message)


else:
    st.session_state["messages"] = []
