from typing import Annotated
from typing_extensions import TypedDict
import streamlit as st
from langgraph.graph.message import add_messages

#. State
class State(TypedDict):
    messages: Annotated[list, add_messages]
