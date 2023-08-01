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

import streamlit as st
import json

### import GCP relevant libs
import vertexai
from vertexai.preview.language_models import ChatModel

# initialize vertexai
vertexai.init(project="wizardry-45677", location="us-central1")

def generate_response(input):
    # start chat session,
    chat_model = ChatModel.from_pretrained("chat-bison@001")
    # define parameters
    parameters = {
        "temperature": temperature,
        "max_output_tokens": token_limit,
        "top_p": top_p,
        "top_k": top_k
    }
    chat = chat_model.start_chat()
    response = chat.send_message(input, **parameters)
    return response
    

# create the chatbot interface

st.title(":green[chat-bison@001] 🤓")

temperature = st.sidebar.slider("Temperature",.1,1.0,.2,.1)
token_limit = st.sidebar.slider("Token limit",1,1024,256,1)
top_k = st.sidebar.slider("Top-K",1,40,40,1)
top_p = st.sidebar.slider("Top-P",.0,.1,.8,.1)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)
    response = generate_response(prompt)
    # response = ChatModel.from_pretrained("chat-bison@001").start_chat().send_message(prompt, temperature=temperature, max_output_tokens=token_limit, top_k=top_k,top_p=top_p)
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})