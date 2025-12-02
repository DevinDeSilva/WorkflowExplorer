#!/usr/bin/env python

import yaml
import json
import os
import re
import logging
import pandas as pd
import random
import numpy as np
from dotenv import load_dotenv
from functools import partial
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import dycomutils as common_utils
import time

import json
import logging
import os
import pickle
from collections import defaultdict
from typing import List, Dict, Any, Optional, Set, Tuple, DefaultDict
from tqdm import tqdm

# Import rdflib for parsing the RDF graph
import rdflib
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS

import time


def time_wrapper(func):
    """
    A decorator that prints the execution time of the function it decorates.
    """
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        log.info(f"Function {func.__name__} Took {total_time:.4f} seconds")
        return result
    return wrapper


def is_inv_rel(rel: str) -> bool:
    """Check if a relation is an inverse relation."""
    return rel.endswith("#R")


def get_inv_rel(rel: str) -> str:
    """Get the inverse of a relation, or vice-versa."""
    if is_inv_rel(rel):
        return rel[:-2]  # Remove '#R'
    return f"{rel}#R"


def get_readable_class(cls: str, schema: Optional[Dict[str, Any]] = None) -> str:
    """Get a readable name for a class."""
    if schema and cls in schema["classes"] and "description" in schema["classes"][cls]:
        return schema["classes"][cls]["description"]
    return cls.split(".")[-1]


def get_readable_relation(rel: str, schema: Optional[Dict[str, Any]] = None) -> str:
    """Get a readable name for a relation."""
    if (
        schema
        and rel in schema["relations"]
        and "description" in schema["relations"][rel]
    ):
        return schema["relations"][rel]["description"]
    return rel.split(".")[-1]


def get_reverse_relation(rel: str, schema: Dict[str, Any]) -> Optional[str]:
    """Get the reverse relation from the schema."""
    return schema["relations"].get(rel, {}).get("reverse")


def get_reverse_readable_relation(rel: str, schema: Dict[str, Any]) -> Optional[str]:
    """Get the readable name of the reverse relation."""
    rev_rel = get_reverse_relation(rel, schema)
    if rev_rel and rev_rel in schema["relations"]:
        return schema["relations"][rev_rel].get("description")
    return None


def get_nodes_by_class(
    nodes: List[Dict[str, Any]], cls: str, except_nid: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """Get all nodes of a specific class, with optional exceptions."""
    if except_nid is None:
        except_nid = []
    return [n for n in nodes if n["class"] == cls and n["nid"] not in except_nid]


def get_non_literals(
    nodes: List[Dict[str, Any]], except_nid: Optional[Set[int]] = None
) -> List[Dict[str, Any]]:
    """Get all nodes that are not literals."""
    if except_nid is None:
        except_nid = set()
    return [
        n
        for n in nodes
        if n["nid"] not in except_nid and not n["class"].startswith("type.")
    ]


def legal_class(cls: str) -> bool:
    """Check if a class is a legal starting point (not a literal)."""
    return not cls.startswith("type.")


def legal_relation(rel: str) -> bool:
    """Placeholder for relation filtering logic, if any."""
    # You can add logic here to filter out specific relations
    return True


def graph_query_to_sexpr(*args, **kwargs) -> str:
    """
    Placeholder for the s-expression conversion function.
    In a real scenario, you would copy this function's code here.
    For this example, we'll return a placeholder string.
    """
    # In your actual use, you would copy the full function definition for
    # graph_query_to_sexpr from src.utils.parser
    logging.warning("Using placeholder function for graph_query_to_sexpr")
    return "(PlaceholderSExpression)"


def graph_query_to_sparql(*args, **kwargs) -> str:
    """
    Placeholder for the SPARQL conversion function.
    In a real scenario, you would copy this function's code here.
    For this example, we'll return a placeholder string.
    """
    # In your actual use, you would copy the full function definition for
    # graph_query_to_sparql from src.utils.parser
    logging.warning("Using placeholder function for graph_query_to_sparql")
    return "SELECT ?x WHERE { ?x ?y ?z . } # (Placeholder SPARQL)"

# --- End of Utility Functions ---

# Hard-coded literal_map (as it was an external dependency)
# This maps schema types to their full XSD/RDF URIs
literal_map: Dict[str, str] = {
    "type.string": "http://www.w3.org/2001/XMLSchema#string",
    "type.text": "http://www.w3.org/2001/XMLSchema#string",
    "type.datetime": "http://www.w3.org/2001/XMLSchema#dateTime",
    "type.integer": "http://www.w3.org/2001/XMLSchema#int",
    "type.int": "http://www.w3.org/2001/XMLSchema#int",
    "type.float": "http://www.w3.org/2001/XMLSchema#float",
    "type.boolean": "http://www.w3.org/2001/XMLSchema#boolean",
}

# --- Utility Functions (from original src.utils) ---
# We include these directly to make the script self-contained
# and remove external dependencies.

# --- 2. RDF Helper Functions ---


# --- 1. Setup & Configuration ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
V2_DIR = os.path.join(ROOT_DIR, "v2")

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

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
metadata_path = os.path.join(V2_DIR, "data/workflow/10_sample_graph/chatbs_sample_metadata.json")
log.info(f"Loading metadata: {metadata_path}")
with open(metadata_path, 'r') as f:
    ttl_metadata = json.load(f)


def validate_namespaces(ns: dict):
    """
    Validates the structure of the namespace dictionary.
    R: validate_namespaces
    """
    if not isinstance(ns, dict) or not ns:
        raise ValueError("Namespaces must be a non-empty dictionary.")
    if any(not k for k in ns.keys()):
        raise ValueError("All namespace prefixes must have non-empty names.")
    bad_vals = [k for k, v in ns.items() if not (isinstance(v, str) and v)]
    if bad_vals:
        raise ValueError(f"All namespace IRIs must be non-empty strings. Offenders: {', '.join(bad_vals)}")
    log.debug("Namespaces validated successfully.")
    return True

def make_ttl_namespace(yaml_config: dict) -> dict:
    """
    Creates a namespace dictionary from the YAML config.
    R: make_ttl_namespace
    """
    namespaces = {}
    for item in yaml_config.get('ttl', {}).get('prefixes', []):
        namespaces[item['name']] = item['uri']
    validate_namespaces(namespaces)
    return namespaces

def curie(x: str, ns: dict, default_prefix: str = None, allow_bare: bool = False) -> str:
    """
    Expands a CURIE (e.g., "rdfs:label") into a full IRI.
    R: curie
    """
    if not isinstance(x, str):
        raise TypeError(f"Input must be a string, got {type(x)}")
    
    # R: if (x == "a") ...
    if x == "a":
        return str(RDF.type)
        
    # R: if (grepl("^(https?|urn):", x))
    if x.startswith(("http:", "https:", "urn:")):
        return x
        
    # R: if (grepl(":", x))
    if ":" in x:
        try:
            prefix, local = x.split(":", 1)
        except ValueError:
            raise ValueError(f"Invalid CURIE format: {x}")
            
        if not local:
            raise ValueError(f"Empty local part in CURIE: {x}")
        if prefix not in ns:
            raise ValueError(f"Unknown prefix in CURIE: {x}")
        return ns[prefix] + local
        
    # R: if (!is.null(default_prefix))
    if default_prefix:
        if default_prefix not in ns:
            raise ValueError(f"default_prefix '{default_prefix}' not found in ns")
        return ns[default_prefix] + x
        
    if allow_bare:
        return x
        
    raise ValueError(f"Not a CURIE (no ':') and not a full IRI: {x}")

def reverse_curie(iri: str, ns: dict) -> str:
    """
    Converts a full IRI back to a CURIE using the provided namespaces.
    R: reverse_curie
    """
    for prefix, uri in ns.items():
        if iri.startswith(uri):
            local_part = iri[len(uri):]
            return f"{prefix}:{local_part}"
    return iri  # Return as-is if no matching prefix found

def add_to_graph_func(s: str, p: str, o: str, g: Graph, namespaces: dict, 
                      literal: bool = False, lang: str = None, dtype: str = None):
    """
    Adds a triple to the rdflib Graph, handling CURIE expansion.
    R: add_to_graph
    """
    try:
        s_uri = URIRef(curie(s, namespaces))
        p_uri = URIRef(curie(p, namespaces))
        
        if literal:
            if dtype:
                o_obj = Literal(o, datatype=URIRef(curie(dtype, namespaces)))
            elif lang:
                o_obj = Literal(o, lang=lang)
            else:
                o_obj = Literal(o)
        else:
            o_obj = URIRef(curie(o, namespaces))
            
        g.add((s_uri, p_uri, o_obj))
        
    except Exception as e:
        log.error(f"Failed to add triple: ({s}, {p}, {o}). Error: {e}")

# --- 3. Graph Manager Class (Replaces R's `new.env()`) ---

class GraphManager:
    """
    A class to hold graph state, config, and helper functions,
    replacing the R environment `graph_func`.
    """
    def __init__(self, config: dict, graph_file: str):
        log.info("Initializing GraphManager...")
        self.config = config
        self.graph = Graph()
        self.graph.parse(graph_file, format="turtle")
        log.info(f"Graph loaded with {len(self.graph)} triples.")
        
        self.config['namespaces'] = make_ttl_namespace(self.config)
        
        # Create a partial function, same as R's `partial()`
        # R: graph_func$add_to_graph <- partial(add_to_graph, ...)
        self.add_to_graph = partial(add_to_graph_func, 
                                    g=self.graph, 
                                    namespaces=self.config['namespaces'])

        self.reverse_curie = partial(reverse_curie, ns=self.config['namespaces'])

    def query(self, sparql_query: str, **args) -> pd.DataFrame:
        if args.get('add_header_tail', True):
            sparql_query = self.add_sparql_header_tail(sparql_query)

        return query_func(self.graph, sparql_query, **args)
    
    def add_sparql_header_tail(self, txt):
        TEMPLATE_HEADER = """
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
        """
        
        TEMPLATE_TAIL = """
        \n 
        """
        
        return TEMPLATE_HEADER + txt + TEMPLATE_TAIL

# Initialize the graph manager
# R: graph_func$graph <- rdf_parse("QGraph.ttl", format = "turtle")
graph_manager = GraphManager(config, os.path.join(V2_DIR, "data/workflow/50_sample_graph/chatbs_sample.ttl"))


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

def resolve_curie(
    x: str, 
    ns: dict = graph_manager.config['namespaces'], 
    default_prefix: str = None, 
    allow_bare: bool = False) -> str:
    
    """
    Resolves a CURIE or returns the input if it's already a full IRI.
    R: resolve_curie
    """
    try:
        return curie(x, ns, default_prefix, allow_bare)
    except ValueError:
        return x

# --- 4. Query Graph ---

def query_func(g: Graph, sparql_query: str, *args) -> pd.DataFrame:
    """
    Executes a SPARQL query and returns a pandas DataFrame.
    R: query_func
    """
    try:
        # R: query <- sprintf(sparql_query_temp_get_objects, ...)
        if args:
            query = sparql_query % args
        else:
            query = sparql_query
        
        results = g.query(query)
        
        # Convert results to a pandas DataFrame
        # R: return(query_results)
        data = []
        for row in results:
            data.append({str(var): str(val) for var, val in row.asdict().items()})
        
        if not data:
            return pd.DataFrame(columns=[str(v) for v in results.vars])
            
        return pd.DataFrame(data)
    except Exception as e:
        log.error(f"Failed to execute SPARQL query: {e}")
        return pd.DataFrame()


def regex_add_strings(template, **kwargs) -> str:
    """
    Adds strings to a template using regular expressions.
    R: regex_add_strings
    """
    try:
        for key, value in kwargs.items():
            pattern = r"\{" + re.escape(key) + r"\}"
            template = re.sub(pattern, str(value), template)
    
        return template
    except Exception as e:
        log.error(f"Failed to add strings to template: {e}")
        return ""

# Setup basic logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

os.makedirs("tmp/programs", exist_ok=True)

class ExecutableProgram:
    """
    Represents an executable program with its metadata.
    """

    def __init__(
        self,
        program_id: str,
        name: str,
        solves: str,
        description: str,
        input_spec: Dict[str, Any],
        output_spec: Dict[str, Any],
        code: str,
        example_usage:str,
        example_output:pd.DataFrame,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.program_id: str = program_id
        self.name: str = name
        self.description: str = description
        self.input_spec: Dict[str, Any] = input_spec
        self.output_spec: Dict[str, Any] = output_spec
        self.code: str = code
        self.solves: str = solves
        self.example_usage: str = example_usage 
        self.example_output: pd.DataFrame = example_output
        self.tags: List[str] = tags if tags is not None else []
        self.metadata: Dict[str, Any] = metadata if metadata is not None else {}

@time_wrapper
def path_to_graph(path: List[str]) -> Optional[ExecutableProgram]:
        """
        Converts a single path (a list of node URIs) into a graph structure.
        """
        str_representation = "->".join(path)
        if os.path.exists(os.path.join("tmp/programs", str_representation)):
            return common_utils.serialization.load_pickle(os.path.join("tmp/programs", str_representation))
        
        logger.debug(f"Converting path to graph: {str_representation}")

        # Create nodes
        str_query = "SELECT distinct ?value where {\n" 
        for i in range(1, len(path) -1, 2):
            # Resolve CURIE to get full URI for class lookup
            sub = path[i - 1]
            pred = path[i] 
            obj = path[i + 1]
            
            if len(path) == 3:
                str_query += "  <{obj}>"+f" {pred} "+" ?value  .\n"
                break
            
            if i == 1:
                str_query += "  <{obj}>"+f" {pred} "+f"?a{i}  .\n"
            elif i == len(path) - 2:
                str_query += f"  ?a{i-2}"+f" {pred} "+" ?value  .\n"
            else:
                str_query += f"  ?a{i-2}"+f" {pred} "+f"?a{i}  .\n"
        str_query += "}"

        SPARQL_OBJ_CLASS_TEMPLATE = """SELECT DISTINCT ?value WHERE {
                                        ?value a {class_uri} .
                                        }"""

        #get objects for the class path
        query_df = graph_manager.query(
            regex_add_strings(
                SPARQL_OBJ_CLASS_TEMPLATE,
                class_uri=path[0]
            )
        )

        objs = list(set(query_df['value'].to_list()))
        print(len(objs))
        
        example_output = None
        example_query = None
        for obj in random.sample(objs, len(objs) ):
            example_query = regex_add_strings(
                str_query,
                obj=obj
            )

            #print(example_query)
            print("Executing example query...")
            start_t = time.time()
            example_output = graph_manager.query(example_query)
            end_t = time.time()
            print("end execution.")
            print(f"Query took {end_t - start_t} seconds")
            if not example_output.empty:
                break
            
        if example_output is None or example_output.empty or example_query is None:
            logger.debug(f"No results for path: {str_representation}")
            return None

        question = f"What are the values obtained by traversing the path: {str_representation}?"

        p = ExecutableProgram(
            program_id=f"explore_path_{str_representation}",
            name=f"Explore Path {str_representation}",
            description=question,
            input_spec={"obj": "The URI of the starting object."},
            output_spec={"value": "The resulting values from the path traversal."},
            code=graph_manager.add_sparql_header_tail(
                str_query
            ),
            solves=f"What are the values obtained by traversing the path: {str_representation}?",
            example_usage=example_query,
            example_output=example_output.head(10),
            tags=["path-level", *path],
            metadata={
                "path": path
                }
        )
        
        common_utils.serialization.save_pickle(p, os.path.join("tmp/programs", str_representation))
        return p
    
def process_path(path: List[str]):# Only consider paths with at least one edge
    query_graph = path_to_graph(path)
    if query_graph is not None:
        return query_graph, ' -> '.join(path)
    return None, None

PARALLEL = False
class Explorer:
    """
    Loads a pre-built RDF graph and its JSON schema to perform
    random schema-guided traversals (walks).
    """


    def __init__(self, kg_name: str):
        self.kg_name: str = kg_name
        self.schema: Optional[Dict[str, Any]] = None
        self.schema_dr: Dict[str, Tuple[str, str]] = {}
        self.classes: Set[str] = set()

        # In-memory representation of the graph and schema
        self.out_relations_cls: DefaultDict[str, set] = defaultdict(set)
        self.in_relations_cls: DefaultDict[str, set] = defaultdict(set)
        self.cls_2_entid: DefaultDict[str, set] = defaultdict(set)
        self.entid_2_cls_ent: Dict[str, Dict[str, Any]] = {}
        self.literals_by_cls_rel: DefaultDict[Tuple[str, str], set] = defaultdict(set)
        
        self.all_program_Obj = []

    def load_graph_and_schema(
        self,
        schema_fpath: str,
        rdf_fpath: str,
        processed_fpath: Optional[str] = None,
        use_cache: bool = True,
    ):
        """
        Loads the JSON schema and the RDF graph file.
        It builds the in-memory representation needed for exploration.

        Args:
            schema_fpath: Path to the JSON schema file.
            rdf_fpath: Path to the RDF graph file (e.g., .nt, .ttl, .rdf).
            processed_fpath: Path to a .pkl file for caching the processed data.
            use_cache: If True, try to load from processed_fpath if it exists.
        """
 
        if use_cache and processed_fpath and os.path.exists(processed_fpath):
            logger.info(f"Loading cached processed data from {processed_fpath}")
            with open(processed_fpath, "rb") as f:
                processed = pickle.load(f)
                self.schema = processed["schema"]
                self.schema_dr = processed["schema_dr"]
                self.classes = processed["classes"]
                self.out_relations_cls = processed["out_relations_cls"]
                self.in_relations_cls = processed["in_relations_cls"]
                self.cls_2_entid = processed["cls_2_entid"]
                self.entid_2_cls_ent = processed["entid_2_cls_ent"]
                self.literals_by_cls_rel = processed["literals_by_cls_rel"]
            return

        logger.info(f"Processing schema from {schema_fpath}")

        # 1. Load Schema
        with open(schema_fpath, "r") as f:
            self.schema = json.load(f)

        if not self.schema:
            raise ValueError("Schema could not be loaded or is empty.")

        self.classes = set(self.schema.get("classes", {}).keys())

        for rel, rel_obj in self.schema.get("relations", {}).items():
            domain = rel_obj["domain"]
            range_ = rel_obj["range"]

            for dom in domain:
                for ran in range_:
                    self.schema_dr[rel] = (dom, ran)
                    self.out_relations_cls[dom].add(rel)
                    self.in_relations_cls[ran].add(rel)

        logger.info(f"Loading RDF graph from {rdf_fpath}...")

        # 2. Load RDF Graph
        g = Graph()
        try:
            g.parse(rdf_fpath)
        except Exception as e:
            logger.error(f"Failed to parse RDF file {rdf_fpath}: {e}")
            raise

        logger.info(f"Graph loaded with {len(g)} triples. Indexing entities...")

        # 3. Build in-memory indexes from the graph

        # Get RDFS.label, fall back to a common alt
        label_prop = RDFS.label

        # Index Entities and their labels
        for cls in tqdm(self.classes, desc="Indexing entities by class"):
            if cls.startswith("type."):  # Skip literal types
                continue

            try:
                cls_uri = URIRef(cls)
                for ent_uri in g.subjects(RDF.type, cls_uri):
                    if not isinstance(ent_uri, URIRef):
                        continue  # Skip blank nodes

                    ent_id_str = str(ent_uri)
                    self.cls_2_entid[cls].add(ent_id_str)

                    # Get label
                    label_lit = g.value(ent_uri, label_prop)
                    label_str = (
                        str(label_lit) if label_lit else ent_id_str.split("/")[-1]
                    )

                    self.entid_2_cls_ent[ent_id_str] = {"class": cls, "name": label_str}
            except Exception as e:
                logger.warning(f"Error indexing class {cls}: {e}")

        # Index Literals by (domain_class, relation)
        for rel, (domain, range_) in tqdm(
            self.schema_dr.items(), desc="Indexing literals"
        ):
            if not range_.startswith("type."):  # Skip non-literal ranges
                continue

            try:
                domain_uri = URIRef(domain)
                rel_uri = URIRef(rel)

                # Find all subjects of the domain type
                for s_uri in g.subjects(RDF.type, domain_uri):
                    # For each subject, get the literal objects for this relation
                    for o_lit in g.objects(s_uri, rel_uri):
                        if isinstance(o_lit, Literal):
                            self.literals_by_cls_rel[(domain, rel)].add(str(o_lit))
            except Exception as e:
                logger.warning(f"Error indexing literals for relation {rel}: {e}")

        logger.info("Finished processing graph and schema.")

        # 4. Save to cache if path provided
        if use_cache and processed_fpath:
            logger.info(f"Saving processed data to cache at {processed_fpath}")
            try:
                with open(processed_fpath, "wb") as f:
                    pickle.dump(
                        {
                            "schema": self.schema,
                            "schema_dr": self.schema_dr,
                            "classes": self.classes,
                            "out_relations_cls": self.out_relations_cls,
                            "in_relations_cls": self.in_relations_cls,
                            "cls_2_entid": self.cls_2_entid,
                            "entid_2_cls_ent": self.entid_2_cls_ent,
                            "literals_by_cls_rel": self.literals_by_cls_rel,
                        },
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
            except Exception as e:
                logger.warning(f"Failed to write to cache file {processed_fpath}: {e}")
                         
    def explore_object_of_class(self):
        
        # All object of class
        
        SPARQL_TEMPLATE1 = """SELECT DISTINCT ?value WHERE {
                            ?value a <{class_uri}> .
                            }"""

        
        # example  execution of SPARQL query
        class_Name = resolve_curie('provone:Execution')
        run_query = regex_add_strings(SPARQL_TEMPLATE1, class_uri=class_Name)
        
        question_df = graph_manager.query(run_query)
        
        exe1 = ExecutableProgram(
            program_id="explore_object_of_class",
            name="Explore Objects of Class",
            description="Explores objects of a given class in the RDF graph.",
            input_spec={"class_uri": "The URI of the class to explore."},
            output_spec={"objects": "A list of objects belonging to the specified class."},
            code=SPARQL_TEMPLATE1,
            solves="What are all the objects of a given class?",
            example_usage=regex_add_strings(SPARQL_TEMPLATE1, class_uri=class_Name),
            example_output=question_df.head(10),
            tags=["class-level"]
        )
        
        # Get all propertiess of a object
        SPARQL_TEMPLATE2 = """
                PREFIX prov: <http://www.w3.org/ns/prov#>
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

                SELECT DISTINCT ?p ?o ?pe ?po
                        WHERE {
                            # 1. Start with the properties of the target object
                            <{obj_uri}> ?p ?o .
                            
                            # 2. Check if the object is a prov:Collection
                            # This pattern only binds ?isCollection if <{obj_uri}> is a prov:Collection
                            OPTIONAL {
                                <{obj_uri}> rdf:type prov:Collection .
                                BIND(TRUE AS ?isCollection)
                            }
                            
                            OPTIONAL {
                                FILTER (bound(?isCollection))
                                <{obj_uri}> prov:hadMember ?member .
                                ?member ?pe ?po .
                            }
                        }"""

        
        # example  execution of SPARQL query
        obj_Name = resolve_curie('http://testwebsite/testProgram#AI_Task-information_extractor')
        run_query = regex_add_strings(SPARQL_TEMPLATE2, obj_uri=obj_Name)
        
        question_df = graph_manager.query(run_query)
        print(question_df.head(10))
        
        exe2 = ExecutableProgram(
            program_id="explore_attr_of_object",
            name="Explore Attributes of Object",
            description="Explores Attributes of a given object in the RDF graph.",
            input_spec={"obj_uri": "The URI of the object to explore."},
            output_spec={"attributes": "A list of attributes belonging to the specified object."},
            code=SPARQL_TEMPLATE2,
            solves="What are all the ATTRIBUTES AND VALUES of a given object?",
            example_usage=regex_add_strings(SPARQL_TEMPLATE2, obj_uri=obj_Name),
            example_output=question_df.head(10),
            tags=["class-level"]
        )
        
        return [exe1, exe2]
    
    def explore_literal_paths(self):
        """
        Generates executable programs for finding objects by property value and
        finding property values for a given object.
        """
        programs = []

        # Program 1: Find object by property value
        find_by_obj_template = """SELECT DISTINCT ?value WHERE {
            <{obj_uri}> <{relation_uri}> ?value .
            }"""
        find_by_prop_template = """SELECT DISTINCT ?value WHERE {
            ?value <{relation_uri}> ?prop .
            FILTER(CONTAINS(STR(?prop), "{prop_value}"))
        }"""
        
        for c in self.classes:
            if not legal_class(c):
                continue
            
            obj_list, relation_list, datatype_list = literal_for_class(
                resolve_curie(c)
                )
            if not obj_list or not relation_list or not datatype_list:
                continue
            
            for data in zip(obj_list, relation_list, datatype_list):
                objs, relations, raw_data = data
                if not legal_relation(relations):
                    continue

                example_output = None
                example_query = None
                for ob in objs:
                    example_query = regex_add_strings(
                            find_by_obj_template,
                            obj_uri=ob,
                            relation_uri=relations
                        )
                    example_output = graph_manager.query(example_query)
                    if example_output.empty:
                        continue
                    else:
                        break
                    
                if example_output is None or example_output.empty or example_query is None:
                    continue
                
                print(regex_add_strings(
                            find_by_obj_template,
                            relation_uri=relations
                        ))
                
                p = ExecutableProgram(
                    program_id="explore_object_of_class {c} | find by object uri value | relation:{relation}".format(
                        c=graph_manager.reverse_curie(c),
                        relation=graph_manager.reverse_curie(relations)
                        ),
                    name="Explore Objects of Class",
                    description="For a given object with uri of class {c}, find all {relation} values.".format(
                        c=graph_manager.reverse_curie(c),
                        relation=graph_manager.reverse_curie(relations)
                        ),
                    input_spec={"obj_uri": "The URI of the object of interest."},
                    output_spec={"relation_uri": "Relation of interest."},
                    code=graph_manager.add_sparql_header_tail(
                        regex_add_strings(
                            find_by_obj_template,
                            relation_uri=relations
                        )
                    ),
                    solves="What are all the objects of a given class?",
                    example_usage=example_query,
                    example_output=example_output.head(10),
                    tags=["object-level", "from-object"]
                )

                programs.append(p)
                
                example_output = None
                example_query = None
                for sub in raw_data:
                    example_query = regex_add_strings(
                            find_by_prop_template,
                            prop_value=sub,
                            relation_uri=relations  
                        )
                    example_output = graph_manager.query(example_query)
                    
                    if example_output.empty:
                        continue
                    else:
                        break
                    
                if example_output is None or example_output.empty or example_query is None:
                    continue
                
                p = ExecutableProgram(
                    program_id="explore_object_of_class {c} | find by prop value | relation:{relation}".format(
                        c=graph_manager.reverse_curie(c),
                        relation=graph_manager.reverse_curie(relations)
                        ),
                    name="Explore Objects of Class",
                    description="For a given prop value of class {c}, find all {relation} prop value of the object.".format(
                        c=graph_manager.reverse_curie(c),
                        relation=graph_manager.reverse_curie(relations)
                        ),
                    input_spec={"prop_uri": "The URI of the prop of interest."},
                    output_spec={"relation_uri": "Relation of interest."},
                    code=graph_manager.add_sparql_header_tail(
                        regex_add_strings(
                            find_by_obj_template,
                            relation_uri=relations
                        )
                    ),
                    solves="What are all the objects of a given class?",
                    example_usage=example_query,
                    example_output=example_output,
                    tags=["object-level", 'from-prop']
                )

                programs.append(p)

        return programs

    def explore_workflow_graph(self):

        # All object of class
        self.all_program_Obj.extend(self.explore_object_of_class())
        print(f"Total programs after object of class: {len(self.all_program_Obj)}")

        # Explore class methods
        self.all_program_Obj.extend(self.explore_literal_paths())
        print(f"Total programs after literal paths: {len(self.all_program_Obj)}")
        
        # Methods to object
        self.all_program_Obj.extend(self.generate_queries_from_paths())
        print(f"Total programs after generating queries from paths: {len(self.all_program_Obj)}")
        
        common_utils.serialization.save_pickle(
            self.all_program_Obj,
            os.path.join(V2_DIR, "data/workflow/explored_programs.pkl")
        )
    
    @time_wrapper
    def breadth_first_search(self, start_class: str, entity_length: int = 7) -> List[List[str]]:
        """
        Performs a schema-aware breadth-first search to find all simple paths.
        This search explores the ontology (schema) rather than the instance data.

        Args:
            start_class_uri: The URI of the starting class (can be a CURIE).

        Returns:
            A list of all simple paths, where each path is a list of class URIs.
        """
        from collections import deque

        if not isinstance(start_class, str) or not start_class.strip():
            raise ValueError("start_class must be a non-empty string.")


        if start_class not in self.classes:
            logger.warning(f"Start class {start_class} not found in the schema.")
            return []

        # The queue will store tuples of (current_class_uri, path_list)
        # The path is stored with original CURIEs/URIs for consistency
        queue = deque([(start_class, [start_class])])
        all_paths = []

        while queue:
            current_class, current_path = queue.popleft()

            # Find neighbors using the pre-indexed schema relations
            
            # 1. Outgoing relations (current_class is the domain)
            if current_class in self.out_relations_cls:
                for relation in self.out_relations_cls[current_class]:
                    # The range is the neighbor class
                    _, neighbor_class = self.schema_dr[relation]
                    if neighbor_class not in [c for c in current_path]:
                        new_path = current_path + [relation] + [graph_manager.reverse_curie(neighbor_class)]
                        all_paths.append(new_path)
                        
                        if len(new_path) // 2 < entity_length:
                            queue.append((neighbor_class, new_path))

            # 2. Incoming relations (current_class is the range)
            # if current_class in self.in_relations_cls:
            #     for relation in self.in_relations_cls[current_class]:
            #         # The domain is the neighbor class
            #         neighbor_class, _ = self.schema_dr[relation]
            #         if neighbor_class not in [resolve_curie(c) for c in current_path]:
            #             new_path = current_path + [graph_manager.reverse_curie(neighbor_class)]
            #             all_paths.append(new_path)
            #             queue.append((neighbor_class, new_path))
        
        logger.info(f"BFS from '{start_class}' found {len(all_paths)} simple schema paths.")
        return all_paths
               
    
    def generate_queries_from_paths(self):
        """
        Generates a graph query for every simple path found via BFS from each class.
        """
            
        collected_graphs = {}           
        for c in self.classes:
            if not legal_class(c):
                continue
            
            # We can start BFS from the class URI itself to find paths in the schema
            paths = self.breadth_first_search(c)
            
            params = {k:[v] for k,v in enumerate(paths)}
            
            if len(params) == 0:
                continue
                
            if  PARALLEL:
                for _,v in common_utils.concurrancy.concurrent_dict_execution(
                    process_path,
                    params= params,
                    num_max_workers=20
                ):
                    if v[0] is not None and v[1] is not None:
                        if v[1] not in collected_graphs:
                            collected_graphs[v[1]] = v[0]
            else:
                for _, path in params.items():
                    query_graph, path_str = process_path(path[0])
                    if query_graph is not None and path_str is not None:
                        if path_str not in collected_graphs:
                            collected_graphs[path_str] = query_graph

        logger.info(f"Generated {len(collected_graphs)} query graphs from all class paths.")
        return list(collected_graphs.values())

if __name__ == "__main__":
    workflow_explorer = Explorer(kg_name="workflow")
    workflow_explorer.load_graph_and_schema(
        schema_fpath=os.path.join(V2_DIR, "data/workflow/schema.json"),
        rdf_fpath=os.path.join(V2_DIR, "data/workflow/10_sample_graph/chatbs_sample.ttl"),
        processed_fpath=os.path.join(V2_DIR, "data/workflow/processed_workflow.pkl"),
        use_cache=True,
    )

    workflow_explorer.explore_workflow_graph()