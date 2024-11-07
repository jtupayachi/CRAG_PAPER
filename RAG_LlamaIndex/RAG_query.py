# -*- coding: utf-8 -*-
import ast
from datetime import datetime
import re
from langchain.agents import initialize_agent, Tool
from datetime import datetime
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
import psycopg2
from sqlalchemy import make_url
from llama_index.vector_stores.postgres import PGVectorStore
from pathlib import Path
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from typing import Any, List
from llama_index.core.schema import NodeWithScore
from typing import Optional
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from sentence_transformers import SentenceTransformer
import os
import utils
import gradio as gr
from typing import Any, List, Optional
import re
import subprocess
import random

# Set the seed for reproducibility
seed_value = 42
random.seed(seed_value)


def extract_curl_commands(input_string):
    # Use regex to find all 'curl' commands and capture the entire command block
    curl_commands = re.findall(r'curl\s.*(?:\\\n\s+.*)*', input_string)
    # Return the list of curl commands, or an empty string if none found
    return curl_commands if curl_commands else ""


def execute_curl_commands(curl_commands: List[str]):
    for command in curl_commands:
        try:
            print(f"Executing: {command}")
            result = subprocess.run(command,
                                    shell=True,
                                    text=True,
                                    capture_output=True)
            print(f"Output: {result.stdout}")
            if result.stderr:
                print(f"Error: {result.stderr}")
        except Exception as e:
            print(f"Failed to execute command '{command}': {str(e)}")


# Configuration (PG DB must have the pg addon installed, data_cqga must be inserted in a table.)
db_name = "DB"
host = "HOST"
password = "PASS"
port = "PORT"
user = "USER"

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

model_path_string = 'convert/gemma-2-9b-it-crag_new.guff'  #Meta-Llama-3-8B-Instruct-Q8_0.gguf
# model_path_string = 'models/gemma-2-9b-it-Q6_K_L.gguf' #Meta-Llama-3-8B-Instruct-Q8_0.gguf

llm = LlamaCPP(
    model_path=model_path_string,
    temperature=
    0.3,  # Lowered for less randomness if your task requires consistency
    max_new_tokens=
    128,  # Set to match the average token count; consider increasing if needed
    context_window=1024,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": 64},  # Increased GPU layers
    verbose=True,
)

vector_store = PGVectorStore.from_params(database=db_name,
                                         host=host,
                                         password=password,
                                         port=port,
                                         user=user,
                                         table_name="CQGA",
                                         embed_dim=384)


class VectorDBRetriever(BaseRetriever):

    def __init__(
        self,
        vector_store: PGVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 1,
    ) -> None:
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_embedding = embed_model.get_query_embedding(
            query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))
            print(nodes_with_scores)

        return nodes_with_scores


# Setup Retriever and Query Engine
retriever = VectorDBRetriever(vector_store,
                              embed_model,
                              query_mode="default",
                              similarity_top_k=1)

query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)

# Global flags to track feedback and store the last response
waiting_for_feedback = False
last_response = ""
last_query = ""  # To store the last user query

# Async chatbot logic
async def chatbot(query: str) -> str:
    # Get a unique timestamp for logging the relation_id
    relation_id = datetime.now().strftime('%Y%m%d%H%M%S%f')

    # Step 1: Retrieve relevant content based on the query
    retrieved_nodes = retriever._retrieve(QueryBundle(query_str=query))
    retrieved_content = " ".join([node.node.text for node in retrieved_nodes
                                  ])  # Combine texts from retrieved nodes

    # Split the input text to isolate the lists
    lists = retrieved_content.split("'] ")

    # Parse the second list
    second_list = ast.literal_eval(lists[1].strip())
    final_output = ""  # Initialize an empty string to store the final big text

    for instru in second_list:
        # Step 2: Use the retrieved content to construct a follow-up question
        follow_up_query = f"Based on the instruction provided:\n{instru}\n\nGenerate Shell (sh) code that fulfills this instruction. Use the parameters and relevant details from the query:\n{query}\n\nEnsure the Shell script addresses all requirements from the instruction and uses any necessary parameters directly from the query."
        response = str(query_engine.query(follow_up_query))
        # Append each response to the final_output
        final_output += response + "\n"  # Add a newline for better readability between responses

    print(final_output)

    return final_output


def store_response(response, relation_id, feedback):
    # Store the response along with the original query
    if feedback == "Yes":
        utils.update_rag(response, vector_store, embed_model, "conversation",
                         relation_id, feedback)
        print("Feedback Yes: Response stored in RAG with the query.")
    else:
        print("Feedback No: Response not stored.")


# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Live Chat with VectorDB Retriever and LLM")

    chatbot_input = gr.Textbox(label="Enter your query",
                               placeholder="Ask something...")
    chatbot_output = gr.Textbox(label="Response")

    reset_button = gr.Button("Reset Conversation",
                             interactive=False)  # Start disabled

    feedback_yes = gr.Button("Yes", interactive=False)
    feedback_no = gr.Button("No", interactive=False)

    relation_id = None

    async def handle_chat(query):
        global relation_id, waiting_for_feedback, last_response, last_query
        relation_id = datetime.now().strftime('%Y%m%d%H%M%S%f')

        last_query = query  # Store the original query
        last_response = await chatbot(query)

        waiting_for_feedback = True
        return last_response, gr.update(
            interactive=True), gr.update(interactive=True), gr.update(
                interactive=True)  # Enable feedback buttons and reset button

    chatbot_input.submit(
        handle_chat,
        inputs=[chatbot_input],
        outputs=[chatbot_output, feedback_yes, feedback_no, reset_button])

    def handle_feedback(feedback):
        global relation_id, waiting_for_feedback
        response = last_response  # Use the stored response
        query = last_query  # Use the last stored query

        if response:
            to_store = "Response: " + str(query) + " --> " + str(response)
            store_response(to_store, relation_id, feedback)

        waiting_for_feedback = False

        return f"Feedback '{feedback}' received.", gr.update(
            interactive=False), gr.update(interactive=False), gr.update(
                interactive=True)  # Enable reset button after feedback

    feedback_yes.click(
        lambda: handle_feedback("Yes"),
        outputs=[chatbot_output, feedback_yes, feedback_no, reset_button])
    feedback_no.click(
        lambda: handle_feedback("No"),
        outputs=[chatbot_output, feedback_yes, feedback_no, reset_button])

    def reset_chat():
        global waiting_for_feedback
        waiting_for_feedback = False
        return "", gr.update(
            interactive=False), gr.update(interactive=False), gr.update(
                interactive=False)  # Disable all buttons

    reset_button.click(
        reset_chat,
        outputs=[chatbot_output, feedback_yes, feedback_no, reset_button])

# Launch the Gradio app
demo.launch(share=False)
