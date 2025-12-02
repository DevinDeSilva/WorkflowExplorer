
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

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

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
        self.curie = partial(curie, ns=self.config['namespaces'])
        
    def bfs_triples(self, start_node: str, max_depth: int = 2, 
                    p_ignore = []) -> Set[Tuple[str, str, str, int]]:
        visited = set()
        queue = [(start_node, 0)]
        triples = set()

        while queue:
            current_node, depth = queue.pop(0)
            if depth > max_depth or current_node in visited:
                continue
            visited.add(current_node)

            for s, p, o in self.graph.triples((URIRef(self.curie(current_node)), None, None)):
                if str(p) in p_ignore:
                    continue
                triples.add((self.reverse_curie(str(s)), self.reverse_curie(str(p)), self.reverse_curie(str(o)), depth))
                if isinstance(o, URIRef):
                    queue.append((self.reverse_curie(str(o)), depth + 1))

            for s, p, o in self.graph.triples((None, None, URIRef(self.curie(current_node)))):
                if str(p) in p_ignore:
                    continue
                triples.add((self.reverse_curie(str(s)), self.reverse_curie(str(p)), self.reverse_curie(str(o)), depth))
                if isinstance(s, URIRef):
                    queue.append((self.reverse_curie(str(s)), depth + 1))

        return triples

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
    
    def save_graph(self, output_file: str):
        self.graph.serialize(destination=output_file, format="turtle")
        log.info(f"Graph saved to {output_file}")
        
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