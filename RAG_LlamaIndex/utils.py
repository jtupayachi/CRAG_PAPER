
import re
import subprocess
from typing import List
import time
from llama_index.llms.llama_cpp import LlamaCPP
import psycopg2
import json
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
import re
from sentence_transformers import SentenceTransformer
import os
import subprocess
from typing import List, Dict
from urllib.parse import urlparse, parse_qs

def count_parameters_uniques(url):
    # Ensure url is a string
    if not isinstance(url, str):
        raise ValueError("The URL should be a string.")
    
    parsed_url = urlparse(url)
    path = parsed_url.path
    
    # Split the path by slashes and count the non-empty segments
    segments = [segment for segment in path.split('/') if segment]
    num_segments = len(segments)
    
    # Check if there are exactly four distinct query parameters (keys)
    if num_segments == 3:
        return [url]
    else:
        return []


def extract_urls(text):
    # Regular expression to match URLs starting with http or https
    url_pattern = r'https?://[^\s]+'
    # Find all matches in the text
    urls = re.findall(url_pattern, text)
    return urls


def extract_curl_commands(input_string):
    # Use regex to find all 'curl' commands in the input string
    curl_commands = re.findall(r'curl .*', input_string)
    
    # Return the list of curl commands, or an empty string if none found
    return curl_commands if curl_commands else ""

def create_curl_get(url):
    return f"curl -X GET {url}"


def execute_curl_commands(curl_commands: List[str]) -> List[Dict[str, str]]:
    """
    Execute a list of curl commands and return the results.

    Args:
    - curl_commands (List[str]): A list of curl commands to execute.

    Returns:
    - List[Dict[str, str]]: A list of dictionaries, each containing 'command', 'output', and 'error' keys.
    """
    results = []
    
    for command in curl_commands:
        try:
            if 'curl' not in curl_commands:
                command=create_curl_get(curl_commands[0])
            
            print(f"Executing: {command}")
            result = subprocess.run(command, shell=True, text=True, capture_output=True)
            
            # Collect the result in a dictionary
            command_result = {
                "command": command,
                "output": result.stdout,
                "error": result.stderr
            }
            
            results.append(command_result)
        
        except Exception as e:
            results.append({
                "command": command,
                "output": "",
                "error": f"Failed to execute command: {str(e)}"
            })
    
    return results



def execute_db(query,conn):
    # Example of how you might delete an entry from a database using an SQL query
    conn.cursor.execute(query)
    conn.connection.commit()
    


def update_rag(text, vector_store, embed_model, category,relation_id,relationship_type):
    # Create a unique node_id based on the current timestamp
    timestamp = int(time.time() * 1000)  # Milliseconds since epoch
    node_id = f"node_{timestamp:013d}"   # Format as a 13-digit ID for uniqueness

    # Create a single node for the entire text
    node = TextNode(text=text)

    # Initialize tags with the given category
    tags = {str(category)}

    # Assign metadata to the node
    node.metadata = {
        # "source": str(category),               # Indicates the source or category
        "relation_id":relation_id,
        "relationship_type":relationship_type,
        "node_id": node_id,                    # Unique identifier for this node based on timestamp
        "length": len(text),                   # Length of the text
        "tags": list(tags),                    # Tags for categorization
    }

    # Compute the embedding for the node's content
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode="all")
    )
    node.embedding = node_embedding

    # Add the node to the vector store
    vector_store.add([node])

