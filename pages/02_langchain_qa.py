# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

### required to run chroma (in memory vector store)
__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
from langchain.document_loaders import GCSDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import VertexAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import VertexAI
from langchain.chat_models import ChatVertexAI

llm = ChatVertexAI()


# def createLoader(bucket):
#     loader = GCSDirectoryLoader(project_name="blaa-bi-in-a-box", bucket=bucket)
#     data = loader.load()
#     return data


# def createSplits(data):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
#     all_splits = text_splitter.split_documents(data)
#     return all_splits


# def createVectorstore(all_splits):
#     embeddings = VertexAIEmbeddings()
#     vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
#     retriever = vectorstore.as_retriever()
#     return retriever


def source_button(url: str, text: str = None, color="#FD504D"):
    st.markdown(
        f"""
    <a href="{url}" target="_self">
        <div style="
            display: inline-block;
            padding: 0.5em 1em;
            color: #FFFFFF;
            background-color: {color};
            border-radius: 3px;
            text-decoration: none;">
            {text}
        </div>
    </a>
    """,
        unsafe_allow_html=True,
    )


def createVectorStore(bucket):
    loader = GCSDirectoryLoader(project_name="blaa-bi-in-a-box", bucket=bucket)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    embeddings = VertexAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
    return vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}, max_tokens_limit=256
    )


def answerQuestion(input, retriever):
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type="map_reduce",
        retriever=retriever,
        return_source_documents=True,
    )
    return qa_chain({"query": input})


st.title("LangChain 🦜🔗")

if "success_flag" not in st.session_state:
    st.session_state.success_flag = False

if "vector_store" not in st.session_state:
    st.session_state.vector_store = []

user_input = st.sidebar.text_input(
    "Specify the GCS bucket containing your documents here:"
)


if user_input and not st.session_state.success_flag:
    if st.sidebar.button("Initialize Vectorstore"):
        with st.spinner("Initializing...please wait"):
            result = createVectorStore(user_input)
            st.session_state.vector_store = result
        st.sidebar.success("Vectorstore successfully initialized!")
        st.session_state.success_flag = True

if st.session_state.success_flag:
    st.sidebar.success("Vectorstore successfully initialized!")
    question = st.text_input("What would you like to know?")
    if st.button("Submit question"):
        answer = answerQuestion(question, st.session_state.vector_store)
        st.markdown(answer["result"])
        no_of_result = 0
        for d in answer["source_documents"]:
            no_of_result += 1
            st.header(f"Chunk number {no_of_result}")
            st.markdown(d.page_content)
            # st.markdown(d.metadata.get("source"))
            source = d.metadata.get("source")
            source_button(source, "click to get directed to the source document")
            st.divider()
        # st.markdown(answer)