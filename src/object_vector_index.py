# %%
import logging
import os
import sys
from collections import defaultdict
import yaml
import json
import os
import re
import logging
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from functools import partial
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import dycomutils as common_utils
from typing import List, Dict, Any, Optional, Set, Tuple, DefaultDict
import openai

sys.path.append("/home/desild/work/research/chatbs/v2")

from src.utils.helpers import setup_logger
from src.utils.parser import graph_query_to_sexpr, is_inv_rel, get_inv_rel, graph_query_to_sparql
from src.utils.kg import get_readable_relation, get_readable_class, get_non_literals, get_nodes_by_class, \
    get_reverse_relation, get_reverse_readable_relation, prune_graph_query, legal_class, legal_relation
from src.utils.arguments import Arguments
from src.utils.sparql import SPARQLUtil, get_freebase_label, get_freebase_literals_by_cls_rel, \
    get_freebase_entid_lbl_by_cls
from src.utils.maps import literal_map

from transformers import set_seed
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

from src.explorer_updates import Explorer, ExecutableProgram
from src.utils.graph_manager import GraphManager, regex_add_strings

# %%
def llm_chat(system_prompt: str, user_prompt: str, model_version: str, structured_output: bool = False) -> str:
    """
    Sends a chat request to an OpenAI-compatible API.
    R: llm_chat
    """
    client = None
    # R: if ((startsWith(model_version, "gpt-")) || (startsWith(model_version, "o1-")))
    if model_version.startswith("gpt-") or model_version.startswith("o1-"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in .env file")
        client = openai.OpenAI(api_key=api_key)
    else:
        # R: base_url = "http://idea-llm-01.idea.rpi.edu:5000/v1/"
        client = openai.OpenAI(
            base_url="http://idea-llm-01.idea.rpi.edu:5000/v1/",
            api_key=os.getenv("LOCAL_LLM_API_KEY", "no-key-needed") # Add your local key to .env if needed
        )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    chat_params = {
        "model": model_version,
        "messages": messages
    }
    
    # R: if (!is.null(structured_output))
    if structured_output:
        log.info("Requesting structured (JSON) output from LLM.")
        # This is the modern way to request JSON from OpenAI
        chat_params["response_format"] = {"type": "json_object"}
        
    try:
        response = client.chat.completions.create(**chat_params)
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        log.error(f"Error in LLM chat: {e}")
        return f"Error: {e}"

def update_answer(system_prompt: str, user_prompt: str, generated_answer: str, error_message: str, model_version: str) -> str:
    """
    Asks the LLM to correct a previous, failed response.
    R: update_answer
    """
    recorrection_template = f"""
    User prompt : {user_prompt}
    Incorrect generated answer : {generated_answer}
    Error message : {error_message}
    Analyze the original user prompt, the incorrect answer, and the error message. Identify where the generated response failed to meet the prompt’s intent. Then, provide a revised answer.
    """
    return llm_chat(system_prompt, recorrection_template, model_version)


def create_timestamp_id(prefix:str):
    """
    Creates a unique identifier based on the current timestamp.
    R: create_timestamp_id
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{prefix}_{timestamp}"

# %%
# --- 1. Setup & Configuration ---
ROOT_DIR = os.path.abspath("/home/desild/work/research/chatbs")
V2_DIR = os.path.join(ROOT_DIR, "v2")
EXPLORED_PROGRAMS_PICKLE = "data/workflow/explored_programs.pkl"

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# %%
# Load .env file from the specified path
# R: load_dot_env("../ChatBS-NexGen/.env")
env_path = os.path.join(V2_DIR, ".env")
log.info(f"Loading .env file from: {env_path}")
load_dotenv(env_path)

# Load YAML config
# R: config <- yaml::read_yaml(...)
config_path = os.path.join(V2_DIR, "prov.config.yaml")
log.info(f"Loading config: {config_path}")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Load JSON metadata
# R: ttl_metadata <- readLines("QGraph_metadata.json")
metadata_path = os.path.join(V2_DIR, "data/workflow/chatbs_sample_metadata.json")
log.info(f"Loading metadata: {metadata_path}")
with open(metadata_path, 'r') as f:
    ttl_metadata = json.load(f)

# %%
graph_manager = GraphManager(config, os.path.join(V2_DIR, "data/workflow/10_sample_graph/chatbs_sample.ttl"))
schema = common_utils.serialization.load_json(os.path.join(V2_DIR, "data/workflow/schema.json"))
definitions = {'class_definitions':schema['classes'], 'relation_definitions':{k:v["description"] for k,v in schema['relations'].items()}}

# %%
import ollama


# %%
sparql_obj_in_class = """SELECT DISTINCT ?value WHERE {
                                        ?value a <{class_uri}> .
                                        }"""
                                        
sparql_get_object_info = """SELECT DISTINCT ?property ?propertyValue WHERE {
                                        <{object_uri}> ?property ?propertyValue .
                                        }"""

# %%

vec_repr = []
obj_id = []
_class = []
_triples = []

for c in schema['classes']:
    objs = graph_manager.query(
        regex_add_strings(sparql_obj_in_class, 
                          class_uri=graph_manager.curie(c))
        )
    
    objs_list = objs["value"].to_list()
    
    for o in tqdm(objs_list):
        obj_info = graph_manager.query(
            regex_add_strings(sparql_get_object_info,
                              object_uri=o)
        )
        
        #print(o)
        triples_o = graph_manager.bfs_triples(o, 1, p_ignore=["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"])
        #print(len(triples))
        triples = "\n".join(["("+",".join(list(map(str, t)))+")" for t in triples_o])
        
        #print(triples)
        
        # str_obj = f"Class: {c} \n Object URI: {o} \n"
        
        # for idx, row in obj_info.iterrows():
        #     prop = graph_manager.reverse_curie(row['property'])
        #     val = row['propertyValue']
        #     str_obj += f" - {prop}: {val}\n"
        
        
        # #print(f"Object: {o}")
        # #print(str_obj)
        
        embed = ollama.embeddings(
            model='nomic-embed-text', 
            prompt=triples
            )
        
        vec_repr.append(embed["embedding"])
        _triples.append(triples_o)
        _class.append(c)
        obj_id.append(o)

# %%
vec_repr = np.array(vec_repr)
print(vec_repr.shape)

common_utils.serialization.save_pickle(
    {'obj_id': obj_id,
     'vec_repr': vec_repr,
     'class': _class,
     'triples': _triples},
    os.path.join(V2_DIR, "data/workflow/10_sample_graph/object_vector_index.pkl")
)

# %%



