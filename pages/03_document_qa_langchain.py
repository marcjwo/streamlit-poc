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


# gs://qa_documents_marcwo


### required to run chroma (in memory vector store)
__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

### required imports

import streamlit as st

### langchain imports

# load
# split
# storage/embed
# retrieve
# output --> use the llm to generate the answer based on whats found through retrieval

## VectorStoreIndexCreator can cut the following down to a few lines

from langchain.document_loaders import GCSDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import VertexAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatVertexAI

project_id = "blaa-bi-in-a-box"
bucket = "streamlit-test-marcwo"
llm = ChatVertexAI()

loader = GCSDirectoryLoader(project_name=project_id, bucket=bucket)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

embeddings = VertexAIEmbeddings()
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)

question = "What is a Data Mesh?"
# relevant_docs = vectorstore.similarity_search(question)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    chain_type="map_reduce",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
)
qa_chain({"query": question})
