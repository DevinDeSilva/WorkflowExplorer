import yaml
import uuid
import datetime
import pandas as pd
from rdflib import Graph, Literal, URIRef, Namespace
from rdflib.namespace import RDF, XSD
import json
import os

import os
import random
import datetime
import re

def get_files_to_check(root, folder, ignore):
    """
    Lists files in root recursively, filtering for .py files 
    and excluding specific utility scripts.
    """
    files_to_return = []
    
    # os.walk simulates list.files(recursive=TRUE)
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            # Skip files in the ignore list
            if f in ignore:
                continue
            
            # 1. Only keep files ending in .py (converted from .R)
            if not f.endswith(".py"):
                continue
            
            # 2. Exclude specific program files (regex match equivalent)
            # You may need to adjust these names if your Python files are named differently
            if "gen_provone" in f: continue
            if "decorator_func" in f: continue
            if "utils" in f: continue
            
            full_path = os.path.join(dirpath, f)
            files_to_return.append(full_path)
            
    return files_to_return

def entity_marking(entity, config):
    """
    Creates a CURIE (Prefix:Name) based on the config program name.
    """
    # Check if config has the required structure
    if 'program' not in config or 'name' not in config['program']:
        raise ValueError("No program name defined in the config")
    
    prefix = config['program']['name']
    
    # R: sprintf("%s:%s", prefix, entity)
    return f"{prefix}:{entity}"

def name_concat(*args):
    """
    Concatenates names using a hyphen separator.
    """
    # R: paste(..., sep = "-")
    return "-".join([str(arg) for arg in args if arg])

def get_unq_id():
    """
    Generates a unique identifier with timestamp and random number.
    Format: id_YYYYMMDDHHMMSS_RAND
    """
    # R: format(Sys.time(), "%Y%m%d%H%M%S")
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # R: sample(1:1000, 1)
    rand_num = random.randint(1, 1000)
    
    return f"id_{timestamp}_{rand_num}"

def get_time_stamp():
    """
    Returns current timestamp in ISO-like format.
    """
    # R: format(Sys.time(), "%Y-%m-%dT%H:%M:%S")
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

# --- Main Class ---

class ProvOneManager:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.graph = Graph()
        self.namespaces = self._make_ttl_namespace()
        
        # Bind namespaces to rdflib graph for cleaner output
        for prefix, uri in self.namespaces.items():
            self.graph.bind(prefix, Namespace(uri))

    def _validate_namespaces(self, ns):
        if not isinstance(ns, dict):
            raise ValueError("Namespaces must be a dictionary.")
        
        if any(not k for k in ns.keys()):
            raise ValueError("All namespace prefixes must have non-empty names.")
        
        bad_vals = [k for k, v in ns.items() if not isinstance(v, str) or not v]
        if bad_vals:
            bad = ", ".join(bad_vals)
            raise ValueError(f"All namespace IRIs must be non-empty single strings. Offenders: {bad}")

    def _make_ttl_namespace(self):
        namespaces = {}
        # Assuming yaml structure matches: ttl -> prefixes -> list of {name, uri}
        if 'ttl' in self.config and 'prefixes' in self.config['ttl']:
            for item in self.config['ttl']['prefixes']:
                namespaces[item['name']] = item['uri']
        
        self._validate_namespaces(namespaces)
        return namespaces

    def curie(self, x, default_prefix=None, allow_bare=False):
        """
        Expands a CURIE (prefix:local) or short name into a full IRI string.
        """
        if not isinstance(x, str):
            raise ValueError(f"Input must be a string, got {type(x)}")

        # Special case: 'a' -> rdf:type
        if x == "a":
            return str(RDF.type)

        # 1) Already a full IRI
        if x.lower().startswith(("http:", "https:", "urn:")):
            return x

        # 2) CURIE with a prefix
        if ":" in x:
            parts = x.split(":", 1)
            prefix = parts[0]
            local = parts[1]
            
            if not local:
                raise ValueError(f"Empty local part in CURIE: {x}")
            if prefix not in self.namespaces:
                raise ValueError(f"Unknown prefix in CURIE: {x}")
            
            return f"{self.namespaces[prefix]}{local}"

        # 3) Bare name
        if default_prefix:
            if default_prefix not in self.namespaces:
                raise ValueError(f"default_prefix '{default_prefix}' not found in ns")
            return f"{self.namespaces[default_prefix]}{x}"

        if allow_bare:
            return x

        raise ValueError(f"Not a CURIE (no ':') and not a full IRI: {x}")

    def add_to_graph(self, s, p, o, literal=False, lang=None, dtype=None):
        """
        Adds a triple to the graph.
        """
        # Expand Subject and Predicate
        s_uri = URIRef(self.curie(s))
        p_uri = URIRef(self.curie(p))

        if literal:
            if dtype:
                # Expand dtype CURIE if needed, e.g., xsd:string
                dtype_uri = URIRef(self.curie(dtype))
                o_node = Literal(o, datatype=dtype_uri)
            elif lang:
                o_node = Literal(o, lang=lang)
            else:
                o_node = Literal(o)
        else:
            # Object is a URI
            o_node = URIRef(self.curie(o))

        self.graph.add((s_uri, p_uri, o_node))

    def add_metadata_to_object(self, object_name, metadata):
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dict")

        for n, value in metadata.items():
            if not isinstance(value, str):
                raise ValueError(f"Metadata values must be strings. Key: {n}")
            
            self.add_to_graph(object_name, n, value, literal=True, lang="en", dtype='xsd:string')

    # --- ProvOne Specific Functions ---

    def prov_program(self, name, has_in_port, has_out_port, has_sub_program=None, metadata=None, ai_task=None):
        print(f"Creating program: {name}")
        if 'program' not in self.config or 'name' not in self.config['program']:
            raise ValueError("No program name defined in the config")

        prog_name = entity_marking(name, self.config)
        
        details = {
            "name": prog_name,
            "hasInPort": has_in_port, # assuming dict structure
            "hasOutPort": has_out_port
        }

        self.add_to_graph(prog_name, "a", "provone:Program")
        
        if ai_task:
            if not isinstance(ai_task, dict):
                raise ValueError("AI tasks must be a dict")
            self.add_to_graph(prog_name, "a", "eo:SystemRecommendation")

        # Process InPorts
        for port_key, port in has_in_port.items():
            port_ident = name_concat(prog_name, port['name'])
            self.add_to_graph(prog_name, "provone:hasInPort", port_ident)
            self.add_to_graph(port_ident, "a", "provone:Port")
            
            if 'default' in port and port['default'] is not None:
                raise NotImplementedError("provone:hasDefault is not implemented yet")
            
            if 'metadata' in port:
                self.add_metadata_to_object(port_ident, port['metadata'])
            
            details['hasInPort'][port_key]['name'] = port_ident
            details['hasInPort'][port_key]['port_key'] = port['name']

        # Process OutPorts
        for port_key, port in has_out_port.items():
            port_ident = name_concat(prog_name, port['name'])
            self.add_to_graph(prog_name, "provone:hasOutPort", port_ident)
            self.add_to_graph(port_ident, "a", "provone:Port")

            if 'default' in port and port['default'] is not None:
                raise NotImplementedError("provone:hasDefault is not implemented yet")

            if 'metadata' in port:
                self.add_metadata_to_object(port_ident, port['metadata'])

            details['hasOutPort'][port_key]['name'] = port_ident
            details['hasOutPort'][port_key]['port_key'] = port['name']

        # Process SubPrograms
        if has_sub_program:
            for sub_program in has_sub_program:
                self.add_to_graph(prog_name, "provone:hasSubProgram", sub_program['name'])
            details['hasSubProgram'] = has_sub_program

        if metadata:
            self.add_metadata_to_object(prog_name, metadata)

        # Process AI Task
        if ai_task:
            print("AI Task Details:")
            ai_task_name = entity_marking(name_concat("AI_Task", name), self.config)
            self.add_to_graph(ai_task_name, "a", "eo:AITask")

            ai_task_input = {}
            if 'input' in ai_task:
                for inp_key, inp_val in ai_task['input'].items():
                    inp_name = entity_marking(name_concat("AI_Task", name, "Input", inp_key), self.config)
                    
                    self.add_to_graph(ai_task_name, "sio:SIO_000230", inp_name)
                    self.add_to_graph(inp_name, "prov:value", inp_val, literal=True, lang="en", dtype='xsd:string')
                    
                    ai_task_input[inp_key] = {
                        "name": inp_name,
                        "value": inp_val,
                        "metadata": {}
                    }

            self.add_to_graph(ai_task_name, "sio:SIO_000229", prog_name)
            self.add_to_graph(prog_name, "eo:generatedBy", ai_task_name)
            details['aiTask'] = ai_task

        return details

    def prov_channel(self, name, connects_to, metadata=None):
        print(f"Creating channel: {name}")
        if 'program' not in self.config or 'name' not in self.config['program']:
            raise ValueError("No program name defined in the config")

        prog_name = entity_marking(name, self.config)
        self.add_to_graph(prog_name, "a", "provone:Channel")

        if connects_to:
            for port in connects_to:
                self.add_to_graph(port['name'], "provone:connectsTo", prog_name)

        if metadata:
            self.add_metadata_to_object(prog_name, metadata)

        return {
            "name": prog_name,
            "connectsTo": connects_to
        }

    # --- Internal Helpers for Execution ---

    def _function_process_exe_literal(self, flow_data, flow, prog, execution_name, direction):
        data_id = get_unq_id()
        data_name = entity_marking(name_concat("Data", data_id, flow), self.config)
        
        # Common Data Node creation
        self.add_to_graph(data_name, "a", "provone:Data")
        self.add_to_graph(data_name, "prov:value", str(flow_data[flow]['value']), literal=True, lang="en", dtype='xsd:string')

        if direction == "input":
            usage_name = entity_marking(name_concat("Usage", data_id, flow), self.config)
            self.add_to_graph(usage_name, "a", "prov:Usage")
            # prog['hasInPort'] is a dict based on earlier logic
            self.add_to_graph(usage_name, "provone:hadInPort", prog['hasInPort'][flow]['name'])
            self.add_to_graph(usage_name, "provone:hadEntity", data_name)
            
            self.add_to_graph(execution_name, "prov:qualifiedUsage", usage_name)
            self.add_to_graph(execution_name, "prov:used", data_name)
        else:
            generation_name = entity_marking(name_concat("Generation", data_id, flow), self.config)
            self.add_to_graph(generation_name, "a", "prov:Generation")
            self.add_to_graph(generation_name, "provone:hadOutPort", prog['hasOutPort'][flow]['name'])
            self.add_to_graph(generation_name, "provone:hadEntity", data_name)
            
            self.add_to_graph(execution_name, "prov:qualifiedGeneration", generation_name)
            self.add_to_graph(data_name, "prov:wasGeneratedBy", execution_name)

        return {"id": data_id, "name": data_name}

    def _function_collection_entry(self, flow_data, flow, prog, execution_name, direction):
        data_id = get_unq_id()
        data_name = entity_marking(name_concat("Collection", data_id, flow), self.config)

        self.add_to_graph(data_name, "a", "prov:Collection")

        if direction == "input":
            usage_name = entity_marking(name_concat("Usage", data_id, flow), self.config)
            self.add_to_graph(usage_name, "a", "prov:Usage")
            self.add_to_graph(usage_name, "provone:hadInPort", prog['hasInPort'][flow]['name'])
            self.add_to_graph(usage_name, "provone:hadEntity", data_name)
            
            self.add_to_graph(execution_name, "prov:qualifiedUsage", usage_name)
            self.add_to_graph(execution_name, "prov:used", data_name)
        else:
            generation_name = entity_marking(name_concat("Generation", data_id, flow), self.config)
            self.add_to_graph(generation_name, "a", "prov:Generation")
            self.add_to_graph(generation_name, "provone:hadOutPort", prog['hasOutPort'][flow]['name'])
            self.add_to_graph(generation_name, "provone:hadEntity", data_name)
            
            self.add_to_graph(execution_name, "prov:qualifiedGeneration", generation_name)
            self.add_to_graph(data_name, "prov:wasGeneratedBy", execution_name)

        return {"id": data_id, "name": data_name}

    def _function_process_exe_list(self, flow_data, flow, prog, execution_name, direction):
        collection_name = self._function_collection_entry(flow_data, flow, prog, execution_name, direction)
        
        data_names_list = []
        for d in flow_data[flow]['value']:
            data_id = get_unq_id()
            data_name = entity_marking(name_concat("Data", data_id, flow), self.config)
            
            self.add_to_graph(data_name, "a", "provone:Data")
            self.add_to_graph(data_name, "prov:value", str(d), literal=True, lang="en", dtype='xsd:string')
            self.add_to_graph(collection_name['name'], "prov:hadMember", data_name)
            
            data_names_list.append({"id": data_id, "name": data_name})
            
        return {"collection": collection_name, "members": data_names_list}

    def _function_process_exe_prov_data(self, flow_data, flow, prog, execution_name, direction, semantic_map=None):
        flow_id = get_unq_id()
        # Assuming value is dict with id/name keys
        data_name = flow_data[flow]['value']['name']

        if direction == "input":
            usage_name = entity_marking(name_concat("Usage", flow_id, flow), self.config)
            self.add_to_graph(usage_name, "a", "prov:Usage")
            self.add_to_graph(usage_name, "provone:hadInPort", prog['hasInPort'][flow]['name'])
            self.add_to_graph(usage_name, "provone:hadEntity", data_name)
            
            self.add_to_graph(execution_name, "prov:qualifiedUsage", usage_name)
            self.add_to_graph(execution_name, "prov:used", data_name)
        else:
            raise ValueError("Output data is not supported for prov-data type in this logic")

        return flow_data[flow]['value']

    def _function_process_exe_df(self, flow_data, flow, prog, execution_name, direction, semantic_map=None):
        df_value = flow_data[flow]['value']
        if not isinstance(df_value, pd.DataFrame):
            raise ValueError("Input data is not a data frame")

        collection_name = self._function_collection_entry(flow_data, flow, prog, execution_name, direction)
        
        if semantic_map is None:
            semantic_map = {}
            for col in df_value.columns:
                print(f"DFColumn:{col}")
                semantic_map[col] = f"DFColumn:{col}"

        data_names_list = []
        for index, row in df_value.iterrows():
            data_id = get_unq_id()
            data_name = entity_marking(name_concat("Data", data_id, flow), self.config)
            
            self.add_to_graph(data_name, "a", "provone:Data")
            self.add_to_graph(collection_name['name'], "prov:hadMember", data_name)
            
            for col in df_value.columns:
                pred = semantic_map.get(col, f"DFColumn:{col}")
                self.add_to_graph(data_name, pred, str(row[col]), literal=True, lang="en", dtype='xsd:string')
            
            data_names_list.append(data_name)

        return {"collection": collection_name, "members": data_names_list}

    def prov_program_execution(self, prog, inputs, outputs, user, semantic_map=None, metadata=None):
        print(f"Creating program execution for: {prog['name']}")
        if not prog.get('name'):
            raise ValueError("No program name defined in the input program object")

        execution_id = get_unq_id()
        execution_name = entity_marking(execution_id, self.config)
        user_name = entity_marking(user, self.config)
        association_name = name_concat(prog['name'], "Association")

        self.add_to_graph(execution_name, "a", "provone:Execution")
        self.add_to_graph(user_name, "a", "prov:Agent")
        self.add_to_graph(association_name, "a", "prov:Association")

        self.add_to_graph(execution_name, "prov:wasAssociatedWith", user_name)
        self.add_to_graph(association_name, "prov:hadPlan", prog['name'])
        self.add_to_graph(association_name, "prov:agent", user_name)
        self.add_to_graph(execution_name, "prov:qualifiedAssociation", association_name)

        # Process Inputs
        records_inputs = {}
        for inp_key, inp_data in inputs.items():
            dt = inp_data.get('data_type')
            if dt == "literal":
                data_name = self._function_process_exe_literal(inputs, inp_key, prog, execution_name, 'input')
            elif dt == "prov-data":
                data_name = self._function_process_exe_prov_data(inputs, inp_key, prog, execution_name, 'input', semantic_map)
            elif dt == "data_frame":
                data_name = self._function_process_exe_df(inputs, inp_key, prog, execution_name, 'input', semantic_map)
            elif dt == "list":
                data_name = self._function_process_exe_list(inputs, inp_key, prog, execution_name, 'input')
            else:
                raise ValueError(f"Unsupported data type: {dt}")
            
            records_inputs[inp_key] = data_name

        # Process Outputs
        records_outputs = {}
        for out_key, out_data in outputs.items():
            dt = out_data.get('data_type')
            if dt == "literal":
                data_name = self._function_process_exe_literal(outputs, out_key, prog, execution_name, 'output')
            elif dt == "data_frame":
                data_name = self._function_process_exe_df(outputs, out_key, prog, execution_name, 'output', semantic_map)
            elif dt == "list":
                data_name = self._function_process_exe_list(outputs, out_key, prog, execution_name, 'output')
            else:
                raise ValueError(f"Unsupported data type: {dt}")
                
            records_outputs[out_key] = data_name

        if metadata:
            self.add_metadata_to_object(execution_name, metadata)

        return {
            "name": execution_name,
            "inputs": records_inputs,
            "outputs": records_outputs,
            "user": user_name
        }

    def prov_make_list(self, entities, metadata=None):
        collection_id = get_unq_id()
        collection_name = entity_marking(collection_id, self.config)

        self.add_to_graph(collection_name, "a", "prov:Collection")
        for ent in entities:
            self.add_to_graph(collection_name, "prov:hadMember", ent)

        return {
            "collection": {
                "name": collection_name,
                "id": collection_id
            },
            "members": entities
        }

    def prov_was_informed_by(self, informed, informing):
        if not informed.get('name') or not informing.get('name'):
            raise ValueError("Both informed and informing must have a name defined")
        
        self.add_to_graph(informed['name'], "prov:wasInformedBy", informing['name'])

    def prov_was_part_of(self, informed, informing):
        if not informed.get('name') or not informing.get('name'):
            raise ValueError("Both informed and informing must have a name defined")
        
        self.add_to_graph(informed['name'], "provone:wasPartOf", informing['name'])

    def add_to_namespace(self, prefix, uri):
        print(f"Adding to namespace: {prefix} {uri}")
        if prefix in self.namespaces:
            return
        
        self.namespaces[prefix] = uri
        self.graph.bind(prefix, Namespace(uri))
        self._validate_namespaces(self.namespaces)

    def save_prov_graph(self):
        print(self.namespaces)
        
        # Save Metadata sidecar
        metadata = {
            "generatedBy": self.config['program']['name'],
            "generatedAt": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "namespaces": self.namespaces
        }
        
        # Assuming config['ttl']['metadata_path'] and config['ttl']['save_path'] exist
        meta_path = self.config['ttl'].get('metadata_path', 'metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        # Save RDF graph
        save_path = self.config['ttl'].get('save_path', 'provenance.ttl')
        self.graph.serialize(destination=save_path, format="turtle")
