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


def decide_probable_entity(
    question: str, class_schema: Dict[str, str], relation_schema: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    """
    Uses an LLM to decide the most probable entity from a list of candidates based on the prompt.
    R: decide_probable_entity
    """

    user_prompt = f"""
    The user's question is: {question}  
    Please generate an answer that conforms to the above format:
    """

    c_schema = "\n".join([f"{k} - {v}" for k, v in class_schema.items()])
    r_schema = "\n".join([f"{k} - {v}" for k, v in relation_schema.items()])

    system_prompt = """
    I am developing a knowledge graph enhanced question answering system.  
    Your task is to extract conditional entities and their types and destination entities and their types from user input questions.  
    Please select the type of entity from the following table:  Each line describes an entity type, in the format of -- entity type (description information) 
    
    #class
    {class_schema}  
    
    #relation
    {relation_schema}
    
    Rules:  
    
    -The conditional entity is the known information provided in the problem; 
    -The target entity is the content that the user wants to query in the problem; 
    - 
    
    Output format: Give me a JSON list of candidate entities.
    
    Example: 
    
    Example1: 
    Input: "what are the ingredients that were suggested by the LLM during the experiment ?" 
    Output: {
        "conditional_entities": [
            {"name": "ingredients", "type": ["provone:Data", "provone:Collection"]},
            {"name": "LLM during the experiment", "type": ["provone:Execution", "provone:Program"]}
        ],
        "destination_entities": Null
    }  
    
    Example2: 
    Input: "what are the sparql_query_extractor program executions that returned Null?" 
    Output: {
        "conditional_entities": [
            {"name": "sparql_query_extractor", "type": ["provone:Program"]},
            {"name": "Null", "type": ["provone:Data"]}
        ],
        "destination_entities": [
            {"name": "executions", "type": ["provone:Execution"]}
        ]
    }  


    """

    system_prompt = regex_add_strings(
        system_prompt, class_schema=c_schema, relation_schema=r_schema
    )

    # print(system_prompt)

    # system_prompt = f"""
    # You are an expert at ontology designing. use the following schema to identify relevant classes.
    # Schema: {json.dumps(schema, indent=2)}

    # Please select the most probable entity that matches the prompt. Respond with the number corresponding to your choice.
    # and return the output as a list of candidate entities in JSON format.
    # """

    response = llm_chat(system_prompt, user_prompt, "gpt-4o")
    return return_json_formatted(response)


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


class QuestionBreakdownAgent:
    """
    Breaks down a question into important classes and relevant sub-questions.
    R: QuestionBreakdownAgent
    """

    def __init__(
        self,
        schema: Dict[str, str],
        definitions: Dict[str, Any],
        ques_info: pd.DataFrame,
    ):
        self.schema = schema
        self.definitions = definitions

        self.USER_PROMPT_POE = """
        Given the complex question: "{question}", break it down into a series of sub-questions using the provided atomic questions.
        Each sub-question should be linked to an atomic question from the provided list.
        Formulate a plan of execution of these questions to at the end achieve the answer to the complex question.
        """

        self.SYSTEM_PROMPT_POE = """
        You are an expert question breakdown agent. Given a complex question, you will break it down guided by provided atomic questions
        Formulate a plan of execution of these questions to at the end achieve the answer to the complex question. External information is 
        obtained by querying a knowledge graph using those atomic questions to which SPARQL queries are available.
        
        You may need to do operations such as filtering, sorting, counting, aggregating, etc., based on the nature of the question. if so provide `Null` 
        as the atomic question used for that step. 
        When possible give multiple possible ways to calculate.

        IN THE FIRST STEP, YOU MUST ALWAYS IDENTIFY A main class to retrive objects from. next steps would be the traversal of the knowledge graph. 
        ensure the path is reachable in the knowledge graph. 

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

        """

        self.EXPL2_POE = """
        QUESTION: "what are the programs that have AI capabilities?"
        PLANS:
        
        ### Plan 1:
        ### Step 1:
        - Sub question: Identify the program in the whole system.
        - Atomic Question Used: "Explores objects of a given class in the RDF graph"
        
        #### Step 2: 
        - Sub-question: What AI task is the output of this program?.
        - Atomic Question Used: "Explores objects of a given class in the RDF graph"

        #### Step 3: 
        - Sub-question: Filter the programs that have AI capabilities.
        - Atomic Question Used: Null

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
        
        # Text
        PLANS:
        ### Plan 1:
        #### Step 1: Identify the main class
        - Sub question: Identify the program in the whole system.
        - Atomic Question Used: "Explores objects of a given class in the RDF graph"
        
        #### Step 2: Count the entities
        - Sub-question: Count the number of programs identified in Step 1.
        - Atomic Question Used: Null

        JSON:
        [{
            "step1": {
                "sub-question": "Identify the program in the whole system",
                "used_atomic_question": "Explores objects of a given class in the RDF graph"
            },
            "step2": {
                "sub-question": "Count the number of programs identified in Step 1",
                "used_atomic_question": "Null"
            }
         }]
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

    def sub_questions(self, question: str) -> Any:
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

    def plan_of_execution(self, question: str) -> Any:
        """
        Generates a plan of execution for the question.
        R: plan_of_execution
        """

        plans = self.sub_questions(question)
        best_plan = None
        best_cost = float("inf")
        for plan in plans:
            plan_cost = 0
            for i, step in enumerate(plan.values()):
                atomic_question = step.get("used_atomic_question", "")
                matched_question = string_closest_match(
                    atomic_question, ques_info["question_lbl"].tolist()
                )
                if matched_question:
                    row = ques_info[ques_info["question_lbl"] == matched_question].iloc[
                        0
                    ]
                    path_len = len(row["entity"]) + len(row["relations"])
                    plan_cost += pow(2.71828, path_len)

                else:
                    raise ValueError("No matching atomic question found.")

            if plan_cost < best_cost:
                best_cost = plan_cost
                best_plan = plan

        self.logs["best_plan"] = best_plan
        self.logs["best_plan_cost"] = best_cost
        return best_plan

    def save_logs(self, filepath: str):
        """
        Saves the logs to a JSON file.
        R: save_logs
        """
        _dir, _ = os.path.split(filepath)
        if not os.path.exists(_dir):
            os.makedirs(_dir)

        with open(filepath, "w") as f:
            json.dump(self.logs, f, indent=2)


# %%
ques_info.head()


# %%
def break_down_question(question: str, schema: Dict, uniq_id: str) -> Any:
    """
    Breaks down a question into its components and adds them to the graph.
    R: break_down_question
    """

    important_classes = decide_probable_entity(
        question, schema["class_definitions"], schema["relation_definitions"]
    )

    if not important_classes:
        log.error("No important classes identified.")
        relevant_questions = ques_info
    else:
        if important_classes["conditional_entities"]:
            relevant_conditional_types = [
                x["type"] for x in important_classes["conditional_entities"]
            ]
        else:
            relevant_conditional_types = []

        if important_classes["destination_entities"]:
            relevant_destination_types = [
                x["type"] for x in important_classes["destination_entities"]
            ]
        else:
            relevant_destination_types = []

        relevant_types = relevant_conditional_types + relevant_destination_types
        relevant_types = list(
            set(item for sublist in relevant_types for item in sublist)
        )

        print(relevant_types)
        if not relevant_types:
            log.error("No relevant types identified from important classes.")
            relevant_questions = ques_info
        else:
            relevant_questions = ques_info.loc[
                ques_info["entity"].apply(
                    lambda x: len(set(x).intersection(set(relevant_types))) > 0
                )
            ]
    relevant_questions = relevant_questions.reset_index(drop=True)
    composer = QuestionBreakdownAgent(schema, definitions, relevant_questions)
    plans = composer.plan_of_execution(question)

    os.makedirs(f"./logs/{uniq_id}", exist_ok=True)
    composer.save_logs(f"./logs/{uniq_id}/plan.json")
    return plans


def answer_question(question: str, unq_id: str = None) -> Dict[str, Any]:
    """
    Answers a question using the important classes and the graph.
    R: answer_question
    """
    plan = break_down_question(question, definitions, unq_id)

    return {"question": question, "plan": plan}


# %%
# dome = answer_question(
#     "what are the programs generated by AI?",
#     )

# %%
import re


def extract_sparql(llm_output: str) -> str | None:
    """
    Extracts a SPARQL query from a markdown code block.

    Args:
        llm_output: The text output from the LLM.

    Returns:
        The extracted SPARQL query as a string, or None if not found.
    """
    # Regex pattern to find content inside ```sparql ... ```
    # re.DOTALL makes '.' match newline characters
    # re.IGNORECASE makes 'sparql' case-insensitive
    pattern = r"```sparql\s*([\s\S]+?)\s*```"

    match = re.search(pattern, llm_output, re.DOTALL | re.IGNORECASE)

    if match:
        # Return the first capturing group (the content)
        return match.group(1).strip()

    return None


class SPARQLAgent:
    """
    Breaks down a question into important classes and relevant sub-questions.
    R: QuestionBreakdownAgent
    """

    def __init__(self, plan: Dict[str, Dict[str, str]], definitions: Dict[str, Any]):
        self.plan = plan["plan"]
        self.main_question = plan["question"]
        self.definitions = definitions
        self.logs = {}

    def get_function_sparql(self, map_id: str):
        SPARQL_query = """
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

        df = graph_manager.query(regex_add_strings(SPARQL_query, map_id=map_id))

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

    def get_sparql_results(
        self,
        sub_question: str,
        sparql_query: str,
        sparql_query_example: str,
        parameter_info: str,
        parameter_descriptions: str,
        atomic_question: str,
        prev_results: List[Dict[str, Any]],
    ) -> pd.DataFrame:
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
        for multiple executions:
        
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
                for k, v in self.definitions["class_definitions"].items()
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
            main_question=self.main_question,
            sub_question=sub_question,
            atomic_question=atomic_question,
            parameters=parameter_info,
            parameter_desc=parameter_descriptions,
            sparql_query=sparql_query,
            sparql_query_example=sparql_query_example[0],
        )

        response = llm_chat(system_prompt, user_prompt, "gpt-4o")
        params = return_json_formatted(response)

        collect = []
        if isinstance(params, dict):
            params = [params]
        for v in params:
            query = regex_add_strings(sparql_query, **v)
            df = graph_manager.query(query)
            collect.append(df)
        return pd.concat(collect)

    def get_llm_results(
        self, sub_question: str, prev_results: List[Dict[str, Any]]
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
            main_question=self.main_question,
            prev_results_text=_result_str,
            sub_question=sub_question,
        )

        response = llm_chat(system_prompt, user_prompt, "gpt-4o")
        response = return_json_formatted(response)
        return response

    def get_llm_results_sparql(
        self,
        sub_question: str,
        prev_results: List[Dict[str, Any]],
        sparql_results: pd.DataFrame,
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

        user_prompt = regex_add_strings(
            user_prompt,
            main_question=self.main_question,
            prev_results_text=_result_str,
            sub_question=sub_question,
            sparql_results=build_DF_verberlize(
                sparql_results, sparql_results.columns.tolist(), " | "
            ),
        )

        response = llm_chat(system_prompt, user_prompt, "gpt-4o")
        response = return_json_formatted(response)
        return response

    def execute_step(self, step_name: str, prev_results: List[Dict[str, Any]]) -> Any:
        """
        Executes a single step of the plan.
        R: execute_step
        """
        self.logs[step_name] = {}
        step = self.plan.get(step_name, {})
        sub_question = step["sub-question"]  # type: ignore

        atomic_question = step["used_atomic_question"]  # type: ignore

        if atomic_question.lower() != "null":
            if len(atomic_question.split("|")) < 3:
                atomic_question = atomic_question.split("|")
                atomic_question = atomic_question[0]

            atomic_question = string_closest_match(
                atomic_question, ques_info["question_lbl"].tolist()
            )
            ques_info_row = ques_info[ques_info["question_lbl"] == atomic_question]
            mapping_id = ques_info_row["mapping"].values[0]

            self.logs[step_name]["mapping_id"] = mapping_id
            sparql_query_info = self.get_function_sparql(mapping_id)
            self.logs[step_name]["sparql_query_info"] = sparql_query_info

            _results = self.get_sparql_results(
                sub_question,
                sparql_query_info["imp"],
                sparql_query_info["example"],
                sparql_query_info["param_map"],
                sparql_query_info["param_desc"],
                atomic_question,
                prev_results,
            )

            self.logs[step_name]["sparql_results"] = _results.to_dict(orient="records")

            # Use LLM to process previous results and current SPARQL results
            _results = self.get_llm_results_sparql(sub_question, prev_results, _results)
        else:
            atomic_question = "null"
            # Use LLM to process previous results
            _results = self.get_llm_results(sub_question, prev_results)

        return {"sub_question": sub_question, "results": _results}

    def execute_plan(self) -> Any:
        """
        Executes the plan of execution.
        R: execute_plan
        """
        results = []
        for step_name, _ in self.plan.items():
            try:
                step_result = self.execute_step(step_name, prev_results=results)
                results.append(step_result)
            except Exception as e:
                log.error(f"Error executing step {step_name}: {e}")
            pass

        summary = self.summarize_results(results)
        self.logs["summary"] = summary
        self.logs["results"] = results
        return summary

    def summarize_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Summarizes the results using an LLM.
        R: summarize_results
        """
        system_prompt = """
        You are an expert question answering agent Do not make up answers if you don't know the answer Say so. given previous results from knowledge graph queries,
        you will summarize them to answer the main question. Further provide the URI of the facts that mainly effected the 
        explanation.
        
        Give answer in a JSON format with two fields:
        - answer: The concise summary answer to the main question.
        - important_facts: A list of URIs of the facts that mainly effected the explanation
        """

        result_str = ""
        for result in results:
            result_str += f"Sub-question: {result['sub_question']}\n"
            result_str += f"Results: {json.dumps(result['results'], indent=2)}\n\n"

        user_prompt = """
        Main Question: {main_question}
        
        #### Previous results:
        {results_text}
        
        Provide a concise summary answer to the main question based on the above results.
        """

        user_prompt = regex_add_strings(
            user_prompt, main_question=self.main_question, results_text=result_str
        )

        response = llm_chat(system_prompt, user_prompt, "gpt-4o")
        response = return_json_formatted(response)
        return response

    def save_logs(self, filepath: str):
        """
        Saves the logs to a JSON file.
        R: save_logs
        """
        _dir, _ = os.path.split(filepath)
        if not os.path.exists(_dir):
            os.makedirs(_dir)

        with open(filepath, "w") as f:
            json.dump(self.logs, f, indent=2)


# %%

# print(os.getcwd())
# dome = common_utils.serialization.load_json("v2/src/logs/question_breakdown_user_question_20251115173101.json")
# dome = {"question":"what are the programs generated by AI?", "plan":dome['best_plan']}

def question_asker_and_loger(question: str, experiment_id: str):
    log_id = create_timestamp_id("user_question")
    log_id = f"{experiment_id}/{log_id}"
    os.makedirs(f"./logs/{log_id}", exist_ok=True)
    
    
    plan = answer_question(question, log_id)
    # %%
    sparql_agent = SPARQLAgent(plan, definitions)
    sparql_agent.execute_plan()
    sparql_agent.save_logs(f"./logs/{log_id}/sparql_execution.json")

EXPERIEMNT_ID = "Individual_Testing"
if __name__ == "__main__":
    question_asker_and_loger("How many overall steps in this pipeline, number of channels should give this answer?", EXPERIEMNT_ID)