# %%
import logging
import os
import sys
from collections import defaultdict
import torch
import yaml
import json
import os
import re
import logging
import pandas as pd
from dotenv import load_dotenv
from dotenv import load_dotenv
from functools import partial
from rdflib import Graph, Literal, URIRef
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import dycomutils as common_utils
from typing import List, Dict, Any, Optional, Set, Tuple, DefaultDict
import openai
import ollama
import random

sys.path.append("/home/desild/work/research/chatbs/v2")

from src.utils.helpers import setup_logger
from src.utils.parser import (
    graph_query_to_sexpr,
    is_inv_rel,
    get_inv_rel,
    graph_query_to_sparql,
)
from src.utils.kg import (
    get_readable_relation,
    get_readable_class,
    get_non_literals,
    get_nodes_by_class,
    get_reverse_relation,
    get_reverse_readable_relation,
    prune_graph_query,
    legal_class,
    legal_relation,
)
from src.utils.arguments import Arguments
from src.utils.sparql import (
    SPARQLUtil,
    get_freebase_label,
    get_freebase_literals_by_cls_rel,
    get_freebase_entid_lbl_by_cls,
)
from src.utils.maps import literal_map

from transformers import set_seed
from tqdm import tqdm
from sentence_transformers.util import semantic_search

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

from src.explorer_updates import Explorer, ExecutableProgram
from src.utils.graph_manager import GraphManager, regex_add_strings


# %%
def llm_chat(
    system_prompt: str,
    user_prompt: str,
    model_version: str,
    structured_output: bool = False,
) -> str:
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
            api_key=os.getenv(
                "LOCAL_LLM_API_KEY", "no-key-needed"
            ),  # Add your local key to .env if needed
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    chat_params = {"model": model_version, "messages": messages}

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


def update_answer(
    system_prompt: str,
    user_prompt: str,
    generated_answer: str,
    error_message: str,
    model_version: str,
) -> str:
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


def create_timestamp_id(prefix: str):
    """
    Creates a unique identifier based on the current timestamp.
    R: create_timestamp_id
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{timestamp}/{prefix}"


RETRIEVE_FUNCTION_TEMPLATE = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX ep: <http://linkedu.eu/dedalo/explanationPattern.owl#>
        PREFIX eo: <https://purl.org/heals/eo#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX dc: <http://purl.org/dc/elements/1.1/>
        PREFIX food: <http://purl.org/heals/food/>
        PREFIX prov: <http://www.w3.org/ns/prov#>
        PREFIX provone: <http://purl.org/provone#>
        PREFIX sio:<http://semanticscience.org/resource/>
        PREFIX cwfo: <http://cwf.tw.rpi.edu/vocab#>
        PREFIX dcterms: <http://purl.org/dc/terms#>
        PREFIX user: <http://testwebsite/testUser#>
        PREFIX DFColumn: <http://testwebsite/testDFColumn#>
        PREFIX fnom: <https://w3id.org/function/vocabulary/mapping#>
        PREFIX fnoi: <hhttps://w3id.org/function/vocabulary/implementation#>
        PREFIX fnoc: <https://w3id.org/function/vocabulary/composition/0.1.0/>
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX dbp: <http://dbpedia.org/property/>
        PREFIX dbt: <http://dbpedia.org/resource/Template:>
        PREFIX ques: <http://atomic_questions.org/>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        PREFIX fno: <https://w3id.org/function/vocabulary/core#>

        SELECT ?imp ?example ?param_map ?param_desc ?return_map ?return_desc
        WHERE {
            <{map_id}> fno:implementation/rdfs:label ?imp ;
                       fno:function ?function ;
                       fno:parameterMapping/fnom:functionParameter/fno:predicate ?param_map ;
                       fno:parameterMapping/fnom:functionParameter/rdfs:label ?param_desc ;
                       fno:returnMapping/fnom:functionOutput/fno:predicate ?return_map ;
                       fno:returnMapping/fnom:functionOutput/rdfs:label ?return_desc .
            ?function fno:executes/rdfs:label ?example.
                       
        }
        
        """


# %%
# --- 1. Setup & Configuration ---
ROOT_DIR = os.path.abspath("/home/desild/work/research/chatbs")
V2_DIR = os.path.join(ROOT_DIR, "v2")

# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
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
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Load JSON metadata
# R: ttl_metadata <- readLines("QGraph_metadata.json")
metadata_path = os.path.join(V2_DIR, "data/workflow/chatbs_sample_metadata.json")
log.info(f"Loading metadata: {metadata_path}")
with open(metadata_path, "r") as f:
    ttl_metadata = json.load(f)

# %%
graph_manager = GraphManager(
    config, os.path.join(V2_DIR, "data/workflow/explored_programs_fno.ttl")
)
schema = common_utils.serialization.load_json(
    os.path.join(V2_DIR, "data/workflow/schema.json")
)
definitions = {
    "class_definitions": schema["classes"],
    "relation_definitions": {
        k: v["description"] for k, v in schema["relations"].items()
    },
}

# %%
"""
?exploration_id dc:title ?question .
    OPTIONAL {
        ?exploration_id prov:wasAssociatedWith ?program .
        ?program fnoc:hasFunctionMapping ?mapping .
        ?mapping fnom:mapsToFunction ?function .
        ?function fnom:hasInputParameter ?input_param .
        ?input_param fnom:parameterType ?cls .
        ?cls rdfs:subClassOf* dbo:Category .
        ?cls rdfs:label ?category_label .
    }
    BIND(GROUP_CONCAT(DISTINCT ?category_label; SEPARATOR="->") AS ?categories)
}GROUP BY ?exploration_id ?question ?categories 
"""

# %%
GET_ALL_QUESTIONS = """

PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX ep: <http://linkedu.eu/dedalo/explanationPattern.owl#>
PREFIX eo: <https://purl.org/heals/eo#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX dc: <http://purl.org/dc/elements/1.1/>
PREFIX food: <http://purl.org/heals/food/>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX provone: <http://purl.org/provone#>
PREFIX sio:<http://semanticscience.org/resource/>
PREFIX cwfo: <http://cwf.tw.rpi.edu/vocab#>
PREFIX dcterms: <http://purl.org/dc/terms#>
PREFIX user: <http://testwebsite/testUser#>
PREFIX DFColumn: <http://testwebsite/testDFColumn#>
PREFIX fnom: <https://w3id.org/function/vocabulary/mapping#>
PREFIX fnoi: <hhttps://w3id.org/function/vocabulary/implementation#>
PREFIX fnoc: <https://w3id.org/function/vocabulary/composition/0.1.0/>
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbp: <http://dbpedia.org/property/>
PREFIX dbt: <http://dbpedia.org/resource/Template:>
PREFIX ques: <http://atomic_questions.org/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX fno: <https://w3id.org/function/vocabulary/core#>
    
SELECT ?mapping ?question_lbl ?paths
WHERE {
    ?mapping a fno:Mapping .
    ?mapping fno:function ?function .
    ?function fno:solves ?question .
    ?function fno:name ?paths .
    ?question fno:name ?question_lbl .
}

"""

ques_info = graph_manager.query(GET_ALL_QUESTIONS)
ques_info["entity"] = ques_info["paths"].apply(
    lambda x: [y.strip() for i, y in enumerate(x.split("->")) if i % 2 == 0]
)
ques_info["relations"] = ques_info["paths"].apply(
    lambda x: [y.strip() for i, y in enumerate(x.split("->")) if i % 2 == 1]
)

def string_closest_match(target: str, candidates: List[str]) -> str:
    """
    Finds the closest matching string from a list of candidates.
    R: string_closest_match
    """
    target_set = set(target.split(" "))
    candidates_set = [set(c.split("|")[0].split(" ")) for c in candidates]
    matches = [
        len(c.intersection(target_set)) / len(c.union(target_set))
        for c in candidates_set
    ]
    _maxv = np.max(matches)
    _argmax = np.argmax(matches)

    # print(target_set)
    # print(candidates_set)
    # print("Matches:", matches)
    # print(candidates[_argmax])
    return candidates[_argmax]

# %%
def build_DF_verberlize(df: pd.DataFrame, cols: List[str], sep: str) -> str:
    """
    Converts a DataFrame column to a formatted string.
    R: build_DF_to_string
    """
    lines = []
    for _, row in df.iterrows():
        col_lines = []
        for col in cols:
            col_lines.append(f"{row[col]}")
        lines.append(f"{sep.join(col_lines)}")
    return "\n".join(lines)


# %%
class_def = pd.DataFrame(definitions["class_definitions"]).T.reset_index()
relation_def = pd.DataFrame(
    {k: {"description": v} for k, v in definitions["relation_definitions"].items()}
).T.reset_index()

class_def.columns = ["class", "description"]
relation_def.columns = ["relation", "description"]

print(build_DF_verberlize(class_def, ["class", "description"], " - "))


# %%
def extract_json_from_markdown(text: str) -> str | None:
    """
    Extracts JSON from a markdown code block.
    R: extract_json_from_markdown_stringr (simulated)
    """
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1)
    return None

def extract_sparql_from_markdown(text: str) -> str | None:
    """
    Extracts JSON from a markdown code block.
    R: extract_json_from_markdown_stringr (simulated)
    """
    match = re.search(r"```sparql\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1)
    return None


def return_json_formatted(model_response: str):
    """
    Parses a JSON string, with retries for markdown blocks.
    R: return_json_formatted
    """
    try:
        # R: tryCatch({ fromJSON(model_response) })
        return json.loads(model_response)
    except json.JSONDecodeError as e:
        log.warning(
            f"Error parsing JSON (layer 1): {e}. Trying to extract from markdown."
        )
        try:
            # R: tryCatch({ ... extract_json ... })
            json_content = extract_json_from_markdown(model_response)
            if json_content:
                return json.loads(json_content)
            else:
                raise ValueError("No JSON markdown content extracted.")
        except Exception as e2:
            # R: ... return(data.frame(question = NA, explanation = NA))
            log.error(f"Error in parsing JSON (layer 2): {e2}")
            # Return a list as the prompt expects, even on failure
            return []
        
def literal_for_class(cls: str) -> Tuple[list, list, list]:
    """
    Given a class name find all the literal relationship of that class and return a Literal with appropriate datatype.
    by sparql query on the graph.
    """
    
    # Example implementation (you may need to adjust based on your graph structure)
    
    query = """SELECT ?obj ?relation ?datatype WHERE {
                ?obj a <{cls}> .
                ?obj ?relation ?datatype .
            }
            """
    
    query = regex_add_strings(query, cls=cls)
    results = graph_manager.query(query)
    if results.empty:
        return ([], [], [])
    results['is_literal'] = results['datatype'].apply(lambda dt: '@' in str(dt))
    literal_rows = results[results['is_literal']]
    literal_rows = literal_rows.groupby('relation').agg({'obj': list, 'datatype': list}).reset_index()
    obj_list = []
    relation_list = []
    datatype_list = []
    if literal_rows.empty:
        return ([], [], [])
    for _, row in literal_rows.iterrows():
        obj_list.append(row['obj'])
        relation_list.append(row['relation'])
        datatype_list.append(row['datatype'])

    return obj_list, relation_list, datatype_list


def dataframe_to_json_recursive(data: Any) -> Any:
    """
    Recursively traverses a nested structure (dict, list, tuple).
    If a pandas DataFrame is found, it is converted to a JSON string 
    using the orient='records' format.

    Args:
        data (Any): The nested dictionary, list, or DataFrame to process.

    Returns:
        Any: The processed structure with DataFrames replaced by JSON strings.
    """
    # Case 1: Pandas DataFrame (Base Case for Conversion)
    if isinstance(data, pd.DataFrame):
        # Convert DataFrame to JSON string (orient='records' = list of dicts)
        return data.to_json(orient='records')

    # Case 2: Dictionary (Recursive Step)
    elif isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            # Recursive call on the value
            new_dict[key] = dataframe_to_json_recursive(value)
        return new_dict

    # Case 3: List or Tuple (Recursive Step)
    elif isinstance(data, (list, tuple)):
        # Apply the function to each item in the list/tuple
        processed_list = [dataframe_to_json_recursive(item) for item in data]
        # Preserve the original type (list or tuple)
        return type(data)(processed_list)

    # Case 4: Other types (Base Case)
    else:
        return data


class ChoiceAgent:
    def __init__(
        self,
        graph_manager: GraphManager,
        class_definitions: str,
        relation_definitions: str,
        graph_paths: str ,
        model_version: str = "gpt-4o",
    ):
        self.graph_manager = graph_manager
        self.class_definitions = class_definitions
        self.relation_definitions = relation_definitions
        self.graph_paths = graph_paths
        self.model_version = model_version
        self.logs: Dict[str, Any] = {}
        
        self.system_prompt = """
        You are an expert question answering agent utilizing knowledge from a knowledge graph. given a question, you will choose the best strategy to answer it from the following options:
        1. SPARQL_QUERY_CREATION : Generate a SPARQL query to retrieve the answer directly from the knowledge graph.
        2. PLANNING : Create a multi-step plan to answer the question using a series of queries and reasoning steps where the functions are predefined as
                      sparql functions, you are given the sparql functions below.
                      
        Guidelines for choosing the strategy:
        - If the question concerns aggreation of large number of information use a SPARQL_QUERY_CREATION.
        - If the question requires multiple reasoning steps or involves complex relationships, choose PLANNING. 
        
        `#class Schema` provide the ontology context for the classes present in the ontology.
        `#relation Schema` provide the ontology context for the relations present in the ontology.
        `#Graph Path s` provide the ontology context for the functions and paths present in the ontology.
        Output format: Give me a single word response either "SPARQL_QUERY_CREATION" or "PLANNING", don't provide any other response.
                      
        #class
        {class_schema}  
        
        #relation
        {relation_schema}
        
        #Graph Paths
        {graph_paths}
        
        Example1:
        Question: "What are the programs generated by AI?"
        Response: "PLANNING"
        
        Example2:
        Question: "How many unique experimental are there in this, this accounts to the number of unique identifies in the executions"
        Response: "SPARQL_QUERY_CREATION"
        
        Example3:
        Question: "How many overall program executions ?"
        Response: "SPARQL_QUERY_CREATION"
        
        Example4:
        Question: "what is the steps in order of execution of the programs?"
        Response: "PLANNING"
        """
        
        self.system_prompt = regex_add_strings(
            self.system_prompt, 
            class_schema=self.class_definitions, 
            relation_schema=self.relation_definitions,
            graph_paths=self.graph_paths
        )
        
    def run(self, question: str) -> str:
        user_prompt = f"""
            The user's question is: {question}  
            Please generate an answer that conforms to the above format:
            """
        response = llm_chat(
            self.system_prompt,
            user_prompt,
            self.model_version
        )
        return response.strip()
            
class PlanningAgent:
    def __init__(
        self,
        graph_manager: GraphManager,
        class_definitions: Dict[str, Any],
        relation_definitions: Dict[str, Any],
        ques_info: pd.DataFrame,
        model_version: str = "gpt-4o",
    ):
        self.graph_manager = graph_manager
        self.class_definitions = class_definitions
        self.relation_definitions = relation_definitions
        self.ques_info = ques_info
        self.model_version = model_version
        self.logs: Dict[str, Any] = {}
        
        self.USER_PROMPT_POE = """
        Given the complex question: "{question}", break it down into a series of sub-questions using the provided atomic questions.
        Each sub-question should be linked to an atomic question from the provided list.
        Formulate a plan of execution of these questions to at the end achieve the answer to the complex question.
        
        
        """

        self.SYSTEM_PROMPT_POE = """
        You are an expert question breakdown agent. Given a complex question, you will break it down guided by provided atomic questions
        Formulate a plan of execution of these questions to at the end achieve the answer to the complex question. External information is 
        obtained by querying a knowledge graph using those atomic questions to which SPARQL queries are available.
        
        Guideline:
        - Always start the plan with either identifying relevant entities (must use question entity and identify object in the question, NOT JUST A CLASS, if so no atomic question is used) or retrieving objects 
        from a classes in the knowledge graph.
        - You must eaither choose an atomic question OR question entity NEVER BOTH in a single step.
        - The middle steps should involve traversing the knowledge graph using atomic questions to reach the desired information.
        - Final step is retrieving the attributes or information needed to answer the complex question.
        - You may need to do operations such as filtering, sorting, counting, aggregating,performed on the results obtained combine theis with a atomic question step. 
          based on the nature of the question.
        - You must further identify all the entities relevant for sub questions this is output classes, grounding entities (add the probable class as well). 
        - Next steps would be the traversal of the knowledge graph. ensure the path is reachable in the knowledge graph. 
        - If the output class in the final step is provone:Data or eo:Object, ensure to retrieve all the literal attributes of the object 
            using the "Explores Attributes of a given object in the RDF graph." atomic question.

        The available atomic questions are provided below in the format of question | path traversed in the knowledge graph:
        {atomic_questions}
        """

        self.EXPL1_POE = """
        QUESTION: "how many programs are in this system?"
        PLANS:
        ### Plan 1:
        ### Step 1:
        - Sub question: Identify the program in the whole system.
        - Atomic Question Used: "Explores objects of a given class in the RDF graph"
        
        #### Step 2: Count the entities
        - Sub-question: Count the number of programs identified in Step 1.
        - Atomic Question Used: Null

        QUESTION: "list the programs in this system?"
        """

        self.EXPL2_POE = """
        Examples:
        
        QUESTION: "what are the programs that have AI capabilities?"
        
        ### Plan:
        ### Step 1:
        - Sub question: What AI task is the output of this program?.
        - Atomic Question Used: "Explores objects of a given class in the RDF graph."
        - Entities-Question: []
        - Classes: [eo:AITask]
        
        #### Step 2: 
        - Sub-question: "What are the programs generated by these AI tasks?".
        - Atomic Question Used: "What is the output program of this AI task?"
        - Entities-Question: []
        - Classes: [provone:Program]
        
        #### Step 3: 
        - Sub-question: What are the attributes of the program objects?.
        - Atomic Question Used: "Explores Attributes of a given object in the RDF graph."
        - Entities-Question: []
        - Classes: [provone:Program]
        
        QUESTION: "what is the steps in order of execution of the programs?"
        
        ### Plan:
        ### Step 1:
        - Sub question: Identify the program in the whole system.
        - Atomic Question Used: "Explores objects of a given class in the RDF graph."
        - Entities-Question: []
        - Classes: [provone:Program]
        
        #### Step 2: 
        - Sub-question: What are the attributes of the program objects, the attribute information such as ID to determine the order of execution?.
        - Atomic Question Used: "Explores Attributes of a given object in the RDF graph."
        - Entities-Question: []
        - Classes: [provone:Program]

        
        QUESTION: "what was the system prompt generated by the system prompt generation function of the used in the execution with id 1_2"
        
        ### Plan:
        ### Step 1:
        - Sub question: Find the executions associated with id 1_2.
        - Atomic Question Used: Null
        - Entities-Question: [provone:Execution with id 1_2]
        - Classes: [provone:Execution]
        
        ### Step 2:
        - Sub question: What is the program associated with the execution with id 1_2, Filter execution associated with system prompt generation function?
        - Atomic Question Used: "What program was planned for this execution?"
        - Entities-Question: []
        - Classes: [provone:Program]
        
        #### Step 3: 
        - Sub-question: What are the generated data for the system prompt generation function executions?
        - Atomic Question Used: "What data was generated by this execution?"
        - Entities-Question: []
        - Classes: [provone:Data]
        
        #### Step 4: 
        - Sub-question: What are the attributes of the data objects?.
        - Atomic Question Used: "Explores Attributes of a given object in the RDF graph."
        - Entities-Question: []
        - Classes: [provone:Data]
        """

        self.USER_PROMPT_FORMAT = """
        Convert the text to a structured format in JSON as shown in the example.
        
        # Text
        {text}
        
        JSON:
        """

        self.SYSTEM_PROMPT_FORMAT = """
        You are a data formatter. Given a text in structured format, you will convert it into JSON.
        The output should be a list of plans, where each plan contains steps with sub-questions and used atomic questions.
        Use the example below to guide your formatting:
        
        The keys should be exactly as shown in the example.
        
        # Text
        ### Plan:
        ### Step 1:
        - Sub question: Find the executions associated with id 1_2.
        - Atomic Question Used: Null
        - Entities-Question: [provone:Execution with id 1_2]
        - Classes: [provone:Execution]
        
        ### Step 2:
        - Sub question: What is the program associated with the execution with id 1_2, Filter execution associated with system prompt generation function, remove the unwanted?
        - Atomic Question Used: "What program was planned for this execution?"
        - Entities-Question: []
        - Classes: [provone:Program]
        
        #### Step 3: 
        - Sub-question: What are the generated data for the system prompt generation function executions?
        - Atomic Question Used: "What data was generated by this execution?"
        - Entities-Question: []
        - Classes: [provone:Data]
        
        #### Step 4: 
        - Sub-question: What are the attributes of the data objects?.
        - Atomic Question Used: "Explores Attributes of a given object in the RDF graph."
        - Entities-Question: []
        - Classes: [provone:Data]

        JSON:
        {
            "step1": {
                "sub-question": "Find the executions associated with id 1_2",
                "atomic_question": Null,
                "question-entities": ["provone:Execution with id 1_2"],
                "classes": ["provone:Execution"]
            },
            "step2": {
                "sub-question": "What is the program associated with the execution with id 1_2, Filter execution associated with system prompt generation function?",
                "atomic_question": "What program was planned for this execution?"
                "question-entities": [],
                "classes": ["provone:Program"]
            },
            "step3": {
                "sub-question": "What are the generated data for the system prompt generation function executions?",
                "atomic_question": "What data was generated by this execution?"
                "question-entities": [],
                "classes": ["provone:Data"]
            },
            "step4": {
                "sub-question": "What are the attributes of the data objects?",
                "atomic_question": "Explores Attributes of a given object in the RDF graph."
                "question-entities": [],
                "classes": ["provone:Data"]
            }
         }
        """
        self.logs = {}
        self.setup_system_prompt(ques_info)
        
    def setup_system_prompt(self, question_df: pd.DataFrame):
        """
        Sets up the system prompt with atomic questions.
        R: setup_system_prompt
        """
        
        atomic_questions_str = build_DF_verberlize(
            question_df, ["question_lbl", "paths"], " | "
        )
        
        self.logs["question_df_avl"] = atomic_questions_str

        self.SYSTEM_PROMPT_POE = regex_add_strings(
            self.SYSTEM_PROMPT_POE, atomic_questions=atomic_questions_str
        )

        self.full_system_prompt_poe = "\n\n".join(
            [self.SYSTEM_PROMPT_POE, self.EXPL1_POE, self.EXPL2_POE]
        )
        
    def run(self, question: str) -> Dict[str, Any]:
        """
        Breaks down the question using important classes and relevant sub-questions.
        R: break_down_question
        """
        self.logs['question'] = question
        self.logs["sub_question"] = {}

        user_prompt = regex_add_strings(self.USER_PROMPT_POE, question=question)

        response = llm_chat(self.full_system_prompt_poe, user_prompt, "gpt-4o")
        self.logs["sub_question"]["sub_questions_response"] = {
            "system_prompt": self.full_system_prompt_poe,
            "user_prompt": user_prompt,
            "response": response,
        }

        user_prompt = regex_add_strings(self.USER_PROMPT_FORMAT, text=response)

        response = llm_chat(self.SYSTEM_PROMPT_FORMAT, user_prompt, "gpt-4o")
        self.logs["sub_question"]["format_response"] = {
            "system_prompt": self.SYSTEM_PROMPT_FORMAT,
            "user_prompt": user_prompt,
            "response": response,
        }

        response = return_json_formatted(response)
        self.logs["sub_question"]["final_response"] = response
        return response
        
class SPARQLExecutorAgent:
    def __init__(
        self,
        plan: Dict[str, Any],
        vector_store: Dict[str, Any],
        graph_manager: GraphManager,
        class_definitions: Dict[str, Any],
        relation_definitions: Dict[str, Any],
        ques_info: pd.DataFrame,
        model_version: str = "gpt-4o",
    ):
        self.plan = plan
        self.vector_store = vector_store
        self.graph_manager = graph_manager
        self.class_definitions = class_definitions
        self.relation_definitions = relation_definitions
        self.ques_info = ques_info
        self.model_version = model_version
        self.logs: Dict[str, Any] = {}
        
        if isinstance(self.plan, list):
            self.plan = self.plan[0]  # Take the first plan if multiple are provided 
    
    def get_function_sparql(self, map_id: str):
        df = self.graph_manager.query(regex_add_strings(RETRIEVE_FUNCTION_TEMPLATE, map_id=map_id))

        return (
            df.groupby(["imp"])
            .agg(
                {
                    "example": lambda x: x.tolist(),
                    "param_map": lambda x: x.tolist(),
                    "param_desc": lambda x: x.tolist(),
                    "return_map": lambda x: x.tolist(),
                    "return_desc": lambda x: x.tolist(),
                }
            )
            .reset_index()
            .to_dict(orient="records")[0]
        )
        
    def link_filter(self, object_desc: str, _class: str, ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Links an object description to entities in the knowledge graph based on class.
        R: link_filter
        """
        
        cls = [x == _class for x in self.vector_store['class']]
        objs = [o for i, o in enumerate(self.vector_store['obj_id']) if cls[i]]
        _trp = [o for i, o in enumerate(self.vector_store['triples']) if cls[i]]
        _vec = [o for i, o in enumerate(self.vector_store['vec_repr']) if cls[i]]
        
        obj_list, relation_list, datatype_list = literal_for_class(
                graph_manager.curie(_class)
                )
        
        if not obj_list or not relation_list or not datatype_list:
            return objs, _trp, cls, _vec
        
        else:
        
            relation_list = "\n".join(list(map(self.graph_manager.reverse_curie, relation_list)))
            
            system_prompt = """
            You are an expert information extractor. given an object description and a sentence of information find the 
            information that is in the sentence that can be used to search for the object in the knowledge graph.
            
            Guidelines:
            - Focus on extracting key attributes that uniquely identify the object.
            - Use the provided class schema to understand relevant attributes.
            - Return a list of attribute values that can be used for searching, in JSON format.
            
            Relations are defined as:
            {relation_list}
            
            Example1:
            Main_question: what was the sparql results generated by the execution of sparql_query_extractor with id 1_2
            Entity Description:provone:Execution with id 1_2\n'
            OUTPUT: [
                {"dcterms:identifier": "1_2"}
            ]
            """
            
            system_prompt = regex_add_strings(
                system_prompt, 
                relation_list=relation_list
            )
            
            user_prompt = f"""
            Main_question: {object_desc}
            """
            
            _resp = llm_chat(system_prompt, user_prompt, self.model_version)
            _resp = return_json_formatted(_resp)
            
            cond: List[bool] = []
            for i, o in enumerate(_trp):
                scond = False
                for r in _resp:
                    for key, value in r.items():  
                        for s_trp in o:
                            if key == s_trp[1] and value in s_trp[2]:
                                scond = True
                                break
                        if scond:
                            break
                    if scond:
                        break
                cond.append(scond)
            
            objs = [o for i, o in enumerate(objs) if cond[i]]
            _trp = [o for i, o in enumerate(_trp) if cond[i]]
            _vec = [o for i, o in enumerate(_vec) if cond[i]]
            
            return objs, _trp, cls, _vec

    def link_object_in_kg(self, object_desc: str, _class: str) -> List[str]:
        
        embed = ollama.embeddings(
            model='nomic-embed-text', 
            prompt=object_desc
            )
        
        objs, _trp, cls, _vec = self.link_filter(object_desc, _class)
        
        
        search_results = semantic_search(
            query_embeddings=torch.Tensor([embed['embedding']]),
            corpus_embeddings=torch.Tensor(_vec),
            top_k=20,
        )
        
        top_result_idx = []
        for idx in search_results[0]:
            top_result_idx.append(objs[idx['corpus_id']])
            
        return top_result_idx
    
    def get_sparql_results(
        self,
        main_question: str,
        sub_question: str,
        sparql_query: str,
        sparql_query_example: str,
        parameter_info: str,
        parameter_descriptions: str,
        atomic_question: str,
        prev_results: List[Dict[str, Any]],
    ) -> Dict[str, pd.DataFrame]:
        """
        Executes a SPARQL query and returns the results as a DataFrame.
        R: get_sparql_results
        """

        _result_str = ""
        for prev_result in prev_results:
            _result_str += f"Sub-question: {prev_result['sub_question']}\n"
            _result_str += (
                f"Results: {json.dumps(prev_result['results'], indent=2)}\n\n"
            )

        system_prompt = """
        You are an expert SPARQL query executor. given a SPARQL query Templete and question infomation,
        you will specify the values to fill in as a JSON object. 
        
        ONLY return the SPARQL JSON object, use the full URI. and provide a list 
        for multiple executions, use the linked entities if provided:
        
        #class
        {class_schema}
        
        #prev_results
        {prev_results_text}
        
        #Example1:
        JSON:
        [
            {"class_uri":"http://purl.org/provone#Execution"},
            {"class_uri":"http://purl.org/provone#Program"}
        ]
        """

        class_def = "\n".join(
            [
                f"{graph_manager.curie(k)}: {v['description']}"
                for k, v in self.class_definitions.items()
            ]
        )
        system_prompt = regex_add_strings(
            system_prompt, class_schema=class_def, prev_results_text=_result_str
        )

        user_prompt = """
        ### Step Information:
        
        Main Question: {main_question}
        Step Sub-Question: {sub_question}
        Step Atomic Question: {atomic_question}
        
        parameters: 
        {parameters}
        
        parameter descriptions: 
        {parameter_desc}
        
        SPARQL Query Template: 
        {sparql_query}
        
        Sparql Query Usage Example:
        {sparql_query_example}
        
        PARAMETERS TO FILL:
        """

        user_prompt = regex_add_strings(
            user_prompt,
            main_question=main_question,
            sub_question=sub_question,
            atomic_question=atomic_question,
            parameters=parameter_info,
            parameter_desc=parameter_descriptions,
            sparql_query=sparql_query,
            sparql_query_example=sparql_query_example[0],
        )

        response = llm_chat(system_prompt, user_prompt, "gpt-4o")
        params = return_json_formatted(response)

        collect = {}
        if isinstance(params, dict):
            params = [params]
        for v in params:
            query = regex_add_strings(sparql_query, **v)
            df = graph_manager.query(query)
            collect["_".join(v.values())] = df
        
        if len(collect) == 0:
            return pd.DataFrame()
        
        return collect
    
    def get_llm_based_results(
        self,
        sub_question: str,
        prev_results: List[Dict[str, Any]],
        main_question: str,
    ) -> Any:
        """
        Uses an LLM to process previous results and answer the sub-question.
        R: get_llm_results
        """
        _result_str = ""
        for prev_result in prev_results:
            _result_str += f"Sub-question: {prev_result['sub_question']}\n"
            _result_str += (
                f"Results: {json.dumps(prev_result['results'], indent=2)}\n\n"
            )

        system_prompt = """
        You are an expert question answering agent. given previous results from knowledge graph queries,
        you will process them to answer the sub-question and also provide the entities important to answer next step.
        
        Give the answer in JSON format with two fields:
        - answer: The answer to the sub-question.
        - important_entities: A list of important entities to consider for the next step.  
        """

        user_prompt = """
        ### Step Information:
        Main Question: 
        {main_question}
        
        #### Previous results:
        {prev_results_text}
        
        Step Sub-Question: 
        {sub_question}  
        
        Results:
        """

        user_prompt = regex_add_strings(
            user_prompt,
            main_question=main_question,
            prev_results_text=_result_str,
            sub_question=sub_question,
        )

        response = llm_chat(system_prompt, user_prompt, self.model_version)
        response = return_json_formatted(response)
        return response
    
    def get_llm_results_sparql(
        self,
        sub_question: str,
        prev_results: List[Dict[str, Any]],
        sparql_results: Dict[str, pd.DataFrame],
        main_question: str,
    ) -> Any:
        """
        Uses an LLM to process previous results and answer the sub-question.
        R: get_llm_results
        """
        _result_str = ""
        for prev_result in prev_results:
            _result_str += f"Sub-question: {prev_result['sub_question']}\n"
            _result_str += (
                f"Results: {json.dumps(prev_result['results'], indent=2)}\n\n"
            )

        system_prompt = """
        You are an expert question answering agent. given previous results from knowledge graph queries,
        you will process them to answer the sub-question.  you are given also the SPARQL results from the current step 
        to help you answer the sub-question and also provide the entities important to answer next step.
        
        Guidlines:
        - Analyze the SPARQL results in the context of the sub-question.
        - If there is a filteration properly filter the results when providing answers.
        
        Give the answer in JSON format with two fields:
        - answer: The answer to the sub-question.
        - important_entities: A list of important entities to consider for the next step.
        """

        user_prompt = """
        ### Step Information:
        Main Question: {main_question}
        
        #### Previous results:
        {prev_results_text}
        
        Step Sub-Question: 
        {sub_question} 
        Step SPARQL Results: 
        {sparql_results} 
        
        Results:
        """
        
        str_sparql = ""
        for k, v in sparql_results.items():
            str_sparql += f"Result for parameter set: {k}\n"
            str_sparql += build_DF_verberlize(v, v.columns.tolist(), ", ")
            str_sparql += "\n\n"

        user_prompt = regex_add_strings(
            user_prompt,
            main_question = main_question,
            prev_results_text=_result_str,
            sub_question=sub_question,
            sparql_results=str_sparql,
        )

        response = llm_chat(system_prompt, user_prompt, "gpt-4o")
        response = return_json_formatted(response)
        return response
    
    def run_step(self, main_question: str, step_name: str, prev_results: List[Dict[str, Any]]) -> Any:
        """
        Executes a single step of the plan.
        R: execute_step
        """
        self.logs[step_name] = {}
        step = self.plan.get(step_name, {})
        sub_question = step["sub-question"]  # type: ignore

        atomic_question = step["atomic_question"]  # type: ignore
        entity_linking = step["question-entities"]  # type: ignore
        classes = step["classes"]
        
        entity_linking = list(set(entity_linking) - set(self.class_definitions.keys()))
        
        if entity_linking and len(entity_linking) > 0:
            class_ = "-".join(classes)
            linked_entities = []
            for entity_desc in entity_linking:
                linked_entity_ids = self.link_object_in_kg(
                    f"""
                    Main_question: {main_question}\n
                    Entity Description:{entity_desc}\n""",
                    class_
                    )
                linked_entities.extend(linked_entity_ids)

            _results = {"linked_entities": linked_entities}
        elif atomic_question:
            if len(atomic_question.split("|")) < 3:
                atomic_question = atomic_question.split("|")
                atomic_question = atomic_question[0]

            atomic_question = string_closest_match(
                atomic_question, self.ques_info["question_lbl"].tolist()
            )
            ques_info_row = self.ques_info[self.ques_info["question_lbl"] == atomic_question]
            mapping_id = ques_info_row["mapping"].values[0]

            self.logs[step_name]["mapping_id"] = mapping_id
            sparql_query_info = self.get_function_sparql(mapping_id)
            self.logs[step_name]["sparql_query_info"] = sparql_query_info

            _results_df = self.get_sparql_results(
                sub_question,
                main_question,
                sparql_query_info["imp"],
                sparql_query_info["example"],
                sparql_query_info["param_map"],
                sparql_query_info["param_desc"],
                atomic_question,
                prev_results,
            )

            #_results_df = _results.to_dict(orient="records")
            
            for k, v in _results_df.items():
                if "p" in v.columns:
                    if "pe" in v.columns:
                        _results_df[k] = v[["pe", "po"]].to_dict(orient="records")
                    else:
                        _results_df[k] = v[["p", "o"]].drop_duplicates().to_dict(orient="records")
            if "Explores Attributes of a given" not in atomic_question:
                _results = self.get_llm_results_sparql(
                        sub_question, 
                        prev_results, 
                        _results_df, 
                        main_question
                    )
            else:
                _results = _results_df
                    
                
        else:
            atomic_question = "null"
            _results = self.get_llm_based_results(sub_question, prev_results, main_question)

        self.logs[step_name]["_results"] = _results
        return {"sub_question": sub_question, "results": _results}
    
    def run(self, main_question: str) -> Any:
        """
        Executes the plan of execution.
        R: execute_plan
        """
        results = []
        for step_name, _ in self.plan.items():
            # try:
            #     step_result = self.run_step(main_question, step_name, prev_results=results)
            #     results.append(step_result)
            # except Exception as e:
            #     log.error(f"Error executing step {step_name}: {e}")
            #     print(e)
            step_result = self.run_step(main_question, step_name, prev_results=results)
            results.append(step_result)

        summary = self.summarize_results(main_question,results)
        self.logs["summary"] = summary
        self.logs["results"] = results
        return summary
    
    def summarize_results(self, main_question: str, results: List[Dict[str, Any]]) -> str:
        """
        Summarizes the results using an LLM.
        R: summarize_results
        """
        system_prompt = """
        You are an expert explaining agent. Your job is to explain to a user the answer and how you came to that answer 
        Do not make up answers if you don't know the answer Say so. given previous results from knowledge graph queries,
        you will summarize them to explain the answer to the main question. Further provide the URI of the facts that mainly effected the 
        explanation.
        
        Provide an Explanation to the question asked in the main question, use the path and the information gathered to explain, be verbose but truthful to the information provided in the context.
        also provide a list of important facts that mainly effected the explanation (A list of URIs of the facts that mainly 
        effected the explanation) , if data is provided explicitly state them in the answer. More verbose/Informative the better.
        if the value of data objects or object recoreds are provided use them in the answer rather than URIs.
        """

        
        result_str = ""
        for result in results:
            
            try:
                str_res = json.dumps(result['results'], indent=2)
            except Exception:
                str_res = result['results'].to_json(orient='records')
                str_res = json.dumps(json.loads(str_res), indent=2)
                
            result_str += f"Sub-question: {result['sub_question']}\n"
            result_str += f"Results: {str_res}\n\n"

        user_prompt = """
        Main Question: {main_question}
        
        #### Previous results:
        {results_text}
        """

        user_prompt = regex_add_strings(
            user_prompt, main_question=main_question, results_text=result_str
        )

        response = llm_chat(system_prompt, user_prompt, "gpt-4o")
        return response

class SPARQLGeneratorAgent:
    def __init__(
        self,
        graph_manager: GraphManager,
        vector_store: Dict[str, Any],
        class_definitions: Dict[str, Any],
        graph_paths: str,
        relation_definitions: Dict[str, Any],
        model_version: str = "gpt-4o",
    ):
        self.graph_manager = graph_manager
        self.class_definitions = class_definitions
        self.relation_definitions = relation_definitions
        self.graph_paths = graph_paths
        self.model_version = model_version
        self.vector_store = vector_store
        self.logs: Dict[str, Any] = {}
        
        for c in self.class_definitions.keys():
            obj_list, relation_list, datatype_list = literal_for_class(
                self.graph_manager.curie(c)
                )
            
            if not obj_list or not relation_list or not datatype_list:
                continue
            
            for data in zip(obj_list, relation_list, datatype_list):
                objs, relations, raw_data = data
                
                if 'literals' not in self.class_definitions[c]:
                    self.class_definitions[c]['literals'] = set()
                
                self.class_definitions[c]['literals'].add(relations)
                
            self.class_definitions[c]['literals'] = list(self.class_definitions[c]['literals'])         
            
        
        self.system_prompt_plan = """
        You are an expert SPARQL query generation planning agent. given a question, you will generate a plan to generate a SPARQL query that will retrieve the answer 
        directly from the knowledge graph. 
        Output format: Give me a grounded text step by step on the kg that I can use to create the SPARQL query.
        
        Graph Schema:
        Classes:
        {class_schema}
        Object Relations:
        {relation_schema}
        
        Graph Paths (Ontology Representation):
        {graph_paths}
        
        """
        
        self.system_prompt_plan = regex_add_strings(
            self.system_prompt_plan,
            class_schema=json.dumps(self.class_definitions, indent=2),
            relation_schema=self.relation_definitions,
            graph_paths=self.graph_paths,
        )
        
        self.system_prompt_gen = """
        You are an expert SPARQL query generation agent. given a question, you will generate a SPARQL query to retrieve the answer directly from the knowledge graph.
        Output format: Give me a SPARQL query that I can directly execute on the graph .
        
        Plan:
        {plan}
        
        Graph Paths (Ontology Representation):
        {graph_paths}
        
        Classes:
        {class_schema}
        
        Examples:
        {examples}
        """
        
        self.system_prompt_gen = regex_add_strings(
            self.system_prompt_gen,
            class_schema=json.dumps(self.class_definitions, indent=2),
            graph_paths=self.graph_paths,
        )
        
    def kbest_sparql_queries(self, question: str, k: int =3) -> Optional[str]:
        
        embed = ollama.embeddings(
            model='nomic-embed-text', 
            prompt=question
            )
        
        
        search_results = semantic_search(
            query_embeddings=torch.Tensor([embed['embedding']]),
            corpus_embeddings=torch.Tensor(self.vector_store['vec_repr']),
            top_k=k,
        )
        top_result_idx = []
        for idx in search_results[0]:
            top_result_idx.append(self.vector_store['obj_id'][idx['corpus_id']])
            
        return "\n\n".join(top_result_idx)
        
    def run(self, question: str) -> str:
        user_prompt = regex_add_strings(
            """
            The user's question is: {question}  
            Please generate an answer that conforms to the above format:
            """,
            question=question,
        )
        response = llm_chat(
            self.system_prompt_plan,
            user_prompt,
            self.model_version
        )
        
        set_sys_prompt_gen = regex_add_strings(
            self.system_prompt_gen,
            plan=response,
            examples=self.kbest_sparql_queries(question, k=3),
        )
        
        response = llm_chat(
            set_sys_prompt_gen,
            user_prompt,
            self.model_version
        )
        
        sparql_query = extract_sparql_from_markdown(response)
        
        if not sparql_query:
            log.error("No SPARQL query extracted from the response.")
            return None
        
        sparql_results = self.graph_manager.query(sparql_query)
        return sparql_results.to_string()
    
    
def question_asker_and_loger(
    question: str, 
    log_file_path: str, 
    schema_path: str = os.path.join(V2_DIR, "data/workflow/schema.json"),
    obj_vector_store_path: str = os.path.join(V2_DIR, "data/workflow/10_sample_graph/object_vector_index.pkl"),
    func_vector_store_path: str = os.path.join(V2_DIR, "data/workflow/function_vector_index.pkl"),
    ):
    """
    Main function to handle question answering and logging.
    R: question_asker_and_loger
    """
    ques_info = graph_manager.query(GET_ALL_QUESTIONS)
    schema = common_utils.serialization.load_json(
        schema_path
    )
    definitions = {
        "class_definitions": schema["classes"],
        "relation_definitions": {
            k: v["description"] for k, v in schema["relations"].items()
        },
    }
    
    obj_store = common_utils.serialization.load_pickle(obj_vector_store_path)
    func_store = common_utils.serialization.load_pickle(func_vector_store_path)
    
    choice_agent = ChoiceAgent(
        graph_manager,
        definitions["class_definitions"],
        definitions["relation_definitions"],
        graph_paths="\n".join(ques_info["paths"].tolist()),
        model_version="gpt-4o",
    )
    
    choice_response = choice_agent.run(question)
    log.info(f"Choice Agent Response: {choice_response}")
    
    if choice_response == "SPARQL_QUERY_CREATION":
        sparql_agent = SPARQLGeneratorAgent(
            graph_manager,
            func_store,
            definitions["class_definitions"],
            "\n".join(ques_info["paths"].tolist()),
            definitions["relation_definitions"],
            model_version="gpt-4o",
        )
        result = sparql_agent.run(question)
    elif choice_response == "PLANNING":
        planning_agent = PlanningAgent(
            graph_manager,
            definitions["class_definitions"],
            definitions["relation_definitions"],
            ques_info,
            model_version="gpt-4o",
        )
        plan = planning_agent.run(question)
        print(plan)
        
        sparql_exe_agent = SPARQLExecutorAgent(
            plan,
            obj_store,
            graph_manager,
            definitions["class_definitions"],
            definitions["relation_definitions"],
            ques_info,
        )
        
        result = sparql_exe_agent.run(question)
    else:
        log.error(f"Invalid choice response: {choice_response}")
        return
    
    os.path.dirname(log_file_path)
    if not os.path.exists(os.path.dirname(log_file_path)):
        os.makedirs(os.path.dirname(log_file_path))
        
    log_data = {
        "question": question,
        "choice_agent": choice_agent.logs,
        "planning_agent": planning_agent.logs if choice_response == "PLANNING" else None,
        "sparql_executor_agent": sparql_exe_agent.logs if choice_response == "PLANNING" else None,
        "sparql_agent": sparql_agent.logs if choice_response == "SPARQL_QUERY_CREATION" else None,
        "final_answer": result,
    }
    
    common_utils.serialization.save_json(dataframe_to_json_recursive(log_data), log_file_path)
    log.info(f"Log saved to {log_file_path}")

random.seed(1710)
np.random.seed(1710)
torch.manual_seed(1710)

if __name__ == "__main__":
    # what was the sparql results that were returned in the execution with id 1_2
    question = "what is the program that uses as input the output of the llm_chat program?"
    log_file_path = os.path.join(V2_DIR, f"logs/v3/{question}.json")
    question_asker_and_loger(question, log_file_path)
    
    # for que in os.listdir(os.path.join(V2_DIR, f"logs/v3_backup")):
    #     if os.path.exists(os.path.join(V2_DIR, f"logs/v3/{que}.json")):
    #         continue
    #     trys = 0
    #     while trys < 3:
    #         try:
    #             que = que.replace(".json", "")
    #             log_file_path = os.path.join(V2_DIR, f"logs/v3/{que}.json")
    #             question_asker_and_loger(que, log_file_path)
    #             break
    #         except Exception as e:
    #             print(f"Error processing {que}: {e}")
    #             trys += 1