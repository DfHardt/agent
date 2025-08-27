from uuid import uuid4
import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

import main
import config as cfg
import tools as my_tools

st.title("Celso")

thread_id = uuid4()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    prompt = {"messages": [
            SystemMessage(content = cfg.Prompts.system_prompt, id = 'sys_prompt'),
            HumanMessage(content = prompt)
        ]}
    
    
    result = main.graph.invoke(prompt, config = {'configurable': {'thread_id': '1'}})
    new_messages = result["messages"]
    prompt['messages'].extend(new_messages)

    last_message = [m.content for m in new_messages if isinstance(m, AIMessage)][-1]

    main.store_feedback(last_message)

    with st.chat_message("assistant"):
        response = st.markdown(last_message)
        st.session_state.messages.append({"role": "assistant", "content": last_message})