library(yaml)
library(purrr)
library(rdflib)

source("./helpers/provone_llm_functions/decorator_func.R")
source("./helpers/provone_llm_functions/utils.R")

config <- yaml::read_yaml("./helpers/provone_llm_functions/prov.config.yaml")

validate_namespaces <- function(ns) {
  if (!is.list(ns) || is.null(names(ns))) stop("Namespaces must be a named list.")
  bad_keys <- names(ns)[!nzchar(names(ns))]
  if (length(bad_keys)) stop("All namespace prefixes must have non-empty names.")
  bad_vals <- vapply(ns, function(u) !is.character(u) || length(u) != 1 || !nzchar(u), logical(1))
  if (any(bad_vals)) {
    bad <- paste(names(ns)[bad_vals], collapse = ", ")
    stop("All namespace IRIs must be non-empty single strings. Offenders: ", bad)
  }
  invisible(TRUE)
}

# Make ttl header
make_ttl_namespace <- function(yaml_config, g) {
    app_name <- yaml_config$program$name
    namespaces <- list()
    for (item in yaml_config$ttl$prefixes){
        namespaces[[item$name]] <- item$uri
    }

    validate_namespaces(namespaces)
    return(namespaces)
}

curie <- function(x, ns, default_prefix = NULL, allow_bare = FALSE) {
  #print(str(x))
  stopifnot(is.character(x), length(x) == 1)
  if (is.list(ns)) ns <- unlist(ns, use.names = TRUE)
  stopifnot(is.character(ns), !is.null(names(ns)))

  # Special case: SPARQL/Turtle shorthand 'a' for rdf:type
  if (x == "a") {
    return("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
  }

  # 1) Already a full IRI
  if (grepl("^(https?|urn):", x, ignore.case = TRUE)) {
    return(x)
  }

  # 2) CURIE with a prefix
  if (grepl(":", x, fixed = TRUE)) {
    parts <- strsplit(x, ":", fixed = TRUE)[[1]]
    prefix <- parts[1]
    local  <- paste(parts[-1], collapse = ":")
    if (!nzchar(local)) stop("Empty local part in CURIE: ", x)
    if (!prefix %in% names(ns)) stop("Unknown prefix in CURIE: ", x)
    return(paste0(ns[[prefix]], local))
  }

  # 3) Bare name
  if (!is.null(default_prefix)) {
    if (!default_prefix %in% names(ns)) {
      stop("default_prefix '", default_prefix, "' not found in ns")
    }
    return(paste0(ns[[default_prefix]], x))
  }

  if (allow_bare) {
    return(x)
  }

  stop("Not a CURIE (no ':') and not a full IRI: ", x)
}



add_to_graph <- function(s, p, o, .g, namespaces, literal = FALSE, lang = NULL, dtype = NULL) {
  #print(paste("Adding to graph:", s, p, o))
  s <- curie(s, namespaces)
  p <- curie(p, namespaces)

  if (!literal) {
    o <- curie(o, namespaces)
  }

  if (literal) {
    # Add language tag if given
    if (!is.null(lang)) {
      o <- paste0(o, "@", lang)   
    }
    # Add datatype if given
    if (!is.null(dtype)) {
      o <- paste0(o, "^^<", dtype, ">")
    }
    rdflib::rdf_add(.g, s, p, o, objectType = "literal")
  } else {
    rdflib::rdf_add(.g, s, p, o)
  }
}


add_metadata_to_object <- function(object_name, metadata, config) {
    if (!is.list(metadata)) {
        stop("Metadata must be a list")
    }

    for (n in names(metadata)) {
        if (!is.character(metadata[[n]]) || length(metadata[[n]]) != 1) {
            stop("Metadata values must be single strings")
        }
        config$add_to_graph(object_name, n, metadata[[n]], literal = TRUE, lang = "en", dtype = 'xsd:string')
    }
}



# Initialize provone functions
provProgram <- function(
    config, 
    name, 
    hasInPort, 
    hasOutPort, 
    hasSubProgram=NULL,
    metadata=list(),
    aiTask=list(),
    ...) {
    # Process the input text and generate the appropriate TTL content
    # You can use the existing functions defined above
    print(paste("Creating program:", name))
    if (is.null(config$program$name)) {
        stop("No program name defined in the config")
    }

    prog_name <- entity_marking(name,config)
    details <- list(
            name = prog_name,
            hasInPort = hasInPort,
            hasOutPort = hasOutPort
        )

    config$add_to_graph(prog_name, "a", "provone:Program")
    if (!is.null(aiTask)) {
        if (!is.list(aiTask)) {
            stop("AI tasks must be a list")
        }
        config$add_to_graph(prog_name, "a", "eo:SystemRecommendation")
    }


    for (port_key in names(hasInPort)) {
        port <- hasInPort[[port_key]]
        port_ident <- name_concat(prog_name, port$name)

        config$add_to_graph(prog_name, "provone:hasInPort", port_ident)
        config$add_to_graph(port_ident, "a", "provone:Port")

        if (!is.null(port$default)) {
            stop("provone:hasDefault is not implemented yet")
        }
        add_metadata_to_object(port_ident, port$metadata, config)

        details$hasInPort[[port_key]]$name <- port_ident
        details$hasInPort[[port_key]]$port_key <- port$name
    }
    for (port_key in names(hasOutPort)) {
        port <- hasOutPort[[port_key]]
        port_ident <- name_concat(prog_name, port$name)

        config$add_to_graph(prog_name, "provone:hasOutPort", port_ident)
        config$add_to_graph(port_ident, "a", "provone:Port")

        if (!is.null(port$default)) {
            stop("provone:hasDefault is not implemented yet")
        }

        add_metadata_to_object(port_ident, port$metadata, config)

        details$hasOutPort[[port_key]]$name <- port_ident
        details$hasOutPort[[port_key]]$port_key <- port$name
    }

    if (!is.null(hasSubProgram)) {
        for (sub_program in hasSubProgram) {
            config$add_to_graph(prog_name, "provone:hasSubProgram", sub_program$name)
        }
        details$hasSubProgram <- hasSubProgram
    }

    if (!is.null(metadata)) {
        if (!is.list(metadata)) {
            stop("Metadata must be a list")
        }
        add_metadata_to_object(prog_name, metadata, config)
    }

    print("AI Task Details:")
    if (!is.null(aiTask)) {
        if (!is.list(aiTask)) {
            stop("AI tasks must be a list")
        }


        ai_task_name <- entity_marking(name_concat("AI_Task", name), config)
        config$add_to_graph(ai_task_name, "a", "eo:AITask")
        #config$add_to_graph(ai_task_name, "a", "provone:Program")

        ai_task_input <- list()
        for (inp in names(aiTask$input)) {
            inp_name <- entity_marking(name_concat("AI_Task", name, "Input", inp), config)

            #config$add_to_graph(inp_name, "a", 'eo:ObjectRecord')
            config$add_to_graph(ai_task_name, "sio:SIO_000230", inp_name)
            config$add_to_graph(inp_name, "prov:value", aiTask$input[[inp]], literal = TRUE, lang = "en", dtype = 'xsd:string')

            ai_task_input[[inp]] <- list(
                name = inp_name,
                value = aiTask$input[[inp]],
                metadata = list()
            )
        }

        # eo:systemRRecommendation
        #ai_task_output <- list()
        #for (out in names(aiTask$output)) {
        #    out_name <- entity_marking(name_concat("AI_Task", name, "Output", out), config)

        #    config$add_to_graph(out_name, "a", 'eo:SystemRecommendation')
        #    config$add_to_graph(out_name, "sio:SIO_000229", ai_task_name)
        #    config$add_to_graph(out_name, "prov:value", aiTask$output[[out]], literal = TRUE, lang = "en")

        #    ai_task_output[[out]] <- list( 
        #        name = out_name,
        #        value = aiTask$output[[out]],
        #        metadata = list()
        #    )
        #}


        config$add_to_graph(ai_task_name, "sio:SIO_000229", prog_name)
        config$add_to_graph(prog_name, "eo:generatedBy", ai_task_name)
        details$aiTask <- aiTask
    }

    return(
        details
    )
}

provChannel <- function(
    config, 
    name, 
    connectsTo, 
    metadata=list(),
    ...) {
    print(paste("Creating channel:", name))
    if (is.null(config$program$name)) {
        stop("No program name defined in the config")
    }

    prog_name <- entity_marking(name,config)
    config$add_to_graph(prog_name, "a", "provone:Channel")

    if(!is.null(connectsTo)) {
        for (port in connectsTo) {
            config$add_to_graph(port$name, "provone:connectsTo", prog_name)
        }
    }

    if (!is.null(metadata)) {
        add_metadata_to_object(prog_name, metadata, config)
    }

    return(
        list(
            name = prog_name,
            connectsTo = connectsTo
        )
    )
}

function_process_exe_literal <- function(flow_data, flow, prog, execution_name, config, direction) {
    # Process literal input or output
    data_id <- get_unq_id()
    data_name <- entity_marking(name_concat("Data", data_id, flow), config)

    if(direction == "input") {
        usage_name <- entity_marking(name_concat("Usage", data_id, flow), config)

        config$add_to_graph(data_name, "a", "provone:Data")
        #config$add_to_graph(data_name, "a", "eo:ObjectRecord")
        config$add_to_graph(data_name, "prov:value", flow_data[[flow]]$value, literal = TRUE, lang = "en", dtype = 'xsd:string')

        config$add_to_graph(usage_name, "a", "prov:Usage")
        config$add_to_graph(usage_name, "provone:hadInPort", prog$hasInPort[[flow]]$name)
        config$add_to_graph(usage_name, "provone:hadEntity", data_name)

        config$add_to_graph(execution_name, "prov:qualifiedUsage", usage_name)
        config$add_to_graph(execution_name, "prov:used", data_name)
    }else{
        data_id <- get_unq_id()
        generation_name <- entity_marking(name_concat("Generation", data_id, flow), config)

        config$add_to_graph(data_name, "a", "provone:Data")
        #config$add_to_graph(data_name, "a", "eo:ObjectRecord")
        config$add_to_graph(data_name, "prov:value", flow_data[[flow]]$value, literal = TRUE, lang = "en", dtype = 'xsd:string')

        config$add_to_graph(generation_name, "a", "prov:Generation")
        config$add_to_graph(generation_name, "provone:hadOutPort", prog$hasOutPort[[flow]]$name)
        config$add_to_graph(generation_name, "provone:hadEntity", data_name)

        config$add_to_graph(execution_name, "prov:qualifiedGeneration", generation_name)
        config$add_to_graph(data_name, "prov:wasGeneratedBy", execution_name)
    }

    return(list(
        id = data_id,
        name = data_name
    ))

}

function_collection_entry <- function(flow_data, flow, prog, execution_name, config, direction) {
    data_id <- get_unq_id()
    data_name <- entity_marking(name_concat("Collection", data_id, flow), config)

    if(direction == "input") {
        usage_name <- entity_marking(name_concat("Usage", data_id, flow), config)

        config$add_to_graph(data_name, "a", "prov:Collection")

        config$add_to_graph(usage_name, "a", "prov:Usage")
        config$add_to_graph(usage_name, "provone:hadInPort", prog$hasInPort[[flow]]$name)
        config$add_to_graph(usage_name, "provone:hadEntity", data_name)

        config$add_to_graph(execution_name, "prov:qualifiedUsage", usage_name)
        config$add_to_graph(execution_name, "prov:used", data_name)
    }else{
        data_id <- get_unq_id()
        generation_name <- entity_marking(name_concat("Generation", data_id, flow), config)

        config$add_to_graph(data_name, "a", "prov:Collection")

        config$add_to_graph(generation_name, "a", "prov:Generation")
        config$add_to_graph(generation_name, "provone:hadOutPort", prog$hasOutPort[[flow]]$name)
        config$add_to_graph(generation_name, "provone:hadEntity", data_name)

        config$add_to_graph(execution_name, "prov:qualifiedGeneration", generation_name)
        config$add_to_graph(data_name, "prov:wasGeneratedBy", execution_name)
    }

    return(list(
        id = data_id,
        name = data_name
    ))
}

function_process_exe_list <- function(flow_data, flow, prog, execution_name, config, direction) {
    # Process list input or output

    collection_name <- function_collection_entry(flow_data, flow, prog, execution_name, config, direction)

    data_names_list = list()
    for (d in flow_data[[flow]]$value){
        data_id <- get_unq_id()
        data_name <- entity_marking(name_concat("Data", data_id, flow), config)

        config$add_to_graph(data_name, "a", "provone:Data")
        #config$add_to_graph(data_name, "a", "eo:ObjectRecord")
        config$add_to_graph(data_name, "prov:value", d, literal = TRUE, lang = "en", dtype = 'xsd:string')
        config$add_to_graph(collection_name$name, "prov:hadMember", data_name)
        data_names_list <- list.append(data_names_list, list(
            id = data_id,
            name = data_name
        ))
    }

    return(list(collection = collection_name, members = data_names_list))
}

function_process_exe_prov_data <- function(flow_data, flow, prog, execution_name, config, direction, semantic_map=NULL) {
    flow_id <- get_unq_id()
    data_id <- flow_data[[flow]]$value$id
    data_name <- flow_data[[flow]]$value$name

    if(direction == "input") {
        usage_name <- entity_marking(name_concat("Usage", flow_id, flow), config)

        config$add_to_graph(usage_name, "a", "prov:Usage")
        config$add_to_graph(usage_name, "provone:hadInPort", prog$hasInPort[[flow]]$name)
        config$add_to_graph(usage_name, "provone:hadEntity", data_name)

        config$add_to_graph(execution_name, "prov:qualifiedUsage", usage_name)
        config$add_to_graph(execution_name, "prov:used", data_name)
    }else{
        stop("Output data is not supported")
    }

    return(flow_data[[flow]]$value)
}

function_process_exe_df <- function(flow_data, flow, prog, execution_name, config, direction, semantic_map=NULL) {
    if(!is.data.frame(flow_data[[flow]]$value)){
        stop("Input data is not a data frame")
    }
    # Process data frame input or output
    collection_name <- function_collection_entry(flow_data, flow, prog, execution_name, config, direction)

    sel_df <- flow_data[[flow]]$value
    if(is.null(semantic_map)) {
        semantic_map <- list()

        for (col in names(sel_df)) {
            #print(paste("Processing column:", col))
            print(sprintf("DFColumn:%s", col))
            semantic_map[[col]] <- sprintf("DFColumn:%s", col)
        }
    }

    data_names_list = list()
    for (i in 1:nrow(sel_df)){
        data_id <- get_unq_id()
        data_name <- entity_marking(name_concat("Data", data_id, flow), config)
        config$add_to_graph(data_name, "a", "provone:Data")
        #config$add_to_graph(data_name, "a", "eo:ObjectRecord")
        config$add_to_graph(collection_name$name, "prov:hadMember", data_name)

        for (col in names(sel_df)) {
            pred <- ifelse(!is.null(semantic_map[[col]]), semantic_map[[col]], sprintf("DFColumn:%s", col))
            config$add_to_graph(data_name, pred, sel_df[i, col], literal = TRUE, lang = "en", dtype = 'xsd:string')
        }

        data_names_list <- list.append(data_names_list, data_name)
    }

    return(list(collection = collection_name, members = data_names_list))
}

provMakeList <- function(
    config,
    entities,
    metadata=list(),
    ...){

    collection_id <- get_unq_id()
    collection_name <- entity_marking(collection_id, config)

    config$add_to_graph(collection_name, "a", "prov:Collection")
    for(ent in entities){
        config$add_to_graph(collection_name, "prov:hadMember", ent)
    }

    return(list(
        collection = list(
            name = collection_name, 
            id = collection_id
        ), 
        members = entities
    ))
}

provProgramExecution <- function(
    config, 
    prog, 
    inputs, 
    outputs, 
    user, 
    semantic_map=NULL, 
    metadata=list(),
    ...) {

    print(paste("Creating program execution for:", prog$name))
    if (is.null(prog$name)) {
        stop("No program name defined in the config")
    }

    execution_id <- get_unq_id()
    execution_name <- entity_marking(execution_id, config)
    user_name <- entity_marking(user, config)
    association_name <- name_concat(prog$name, "Association")

    config$add_to_graph(execution_name, "a", "provone:Execution")
    config$add_to_graph(user_name, "a", "prov:Agent")
    config$add_to_graph(association_name, "a", "prov:Association")

    config$add_to_graph(execution_name, "prov:wasAssociatedWith", user_name)
    config$add_to_graph(association_name, "prov:hadPlan", prog$name)
    config$add_to_graph(association_name, "prov:agent", user_name)
    config$add_to_graph(execution_name, "prov:qualifiedAssociation", association_name)

    recordsInputs <- list()
    #print("inputs:")
    #print(str(inputs))
    for (inp in names(inputs)) {
        if (inputs[[inp]]$data_type == "literal") {
            data_name <- function_process_exe_literal(
                inputs, inp, prog, execution_name, config, 'input'
            )
        } else if (inputs[[inp]]$data_type == "prov-data") {
            data_name <- function_process_exe_prov_data(
                inputs, inp, prog, execution_name, config, 'input'
            )
        } else if (inputs[[inp]]$data_type == "data_frame") {
            data_name <- function_process_exe_df(
                inputs, inp, prog, execution_name, config, 'input', semantic_map
            )
        } else if (inputs[[inp]]$data_type == "list") {
            data_name <- function_process_exe_list(
                inputs, inp, prog, execution_name, config, 'input'
            )
        } else {
            stop("Unsupported data type")
        }

        recordsInputs[[inp]] <- data_name
    }

    recordsOutputs <- list()
    #print("outputs:")
    #print(str(outputs))
    for (out in names(outputs)) {
        if (outputs[[out]]$data_type == "literal") {
            data_name <- function_process_exe_literal(
                outputs, out, prog, execution_name, config, 'output'
            )
        } else if (outputs[[out]]$data_type == "data_frame") {
            data_name <- function_process_exe_df(
                outputs, out, prog, execution_name, config, 'output', semantic_map
            )
        } else if (outputs[[out]]$data_type == "list") {
            data_name <- function_process_exe_list(
                outputs, out, prog, execution_name, config, 'output'
            )
        } else {
            stop("Unsupported data type")
        }

        recordsOutputs[[out]] <- data_name
    }

    if (!is.null(metadata)) {
        add_metadata_to_object(execution_name, metadata, config)
    }

    return(
        list(
            name = execution_name,
            inputs = recordsInputs,
            outputs = recordsOutputs,
            user = user_name
        )
    )
}



provWasInformedBy <- function(config, informed, informing, ...) {
    # Process the input text and generate the appropriate TTL content
    # You can use the existing functions defined above
    if (is.null(informed$name) || is.null(informing$name)) {
        stop("Both informed and informing must have a name defined")
    }

    config$add_to_graph(informed$name, "prov:wasInformedBy", informing$name)
}

provWasPartOf <- function(config, informed, informing, ...) {
    # Process the input text and generate the appropriate TTL content
    # You can use the existing functions defined above
    if (is.null(informed$name) || is.null(informing$name)) {
        stop("Both informed and informing must have a name defined")
    }
    config$add_to_graph(informed$name, "provone:wasPartOf", informing$name)
}

save_prov_graph <- function(.graph, config) {
    print(str(config$namespaces))  

    # metadata
    metadata <- toJSON(
        list(
            generatedBy = config$program$name,
            generatedAt = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
            namespaces = config$namespaces
        ),
        pretty=TRUE
        )

    writeLines(metadata, con = config$ttl$metadata_path)

    # Save the RDF graph to a file
        rdf_serialize(
            .graph,
            config$ttl$save_path,
            format    = "turtle",
            namespace = config$namespaces,
            prefix = names(config$namespaces)
        )
}

add_to_namespace <- function(prefix, uri, env) {
    print(paste("Adding to namespace:", prefix, uri))
    if (prefix %in% names(env$config$namespaces)) {
        return(invisible(TRUE))
    }

    env$config$namespaces[[prefix]] <- uri
    validate_namespaces(env$config$namespaces)
    return(invisible(TRUE))
}

prov_module_func <- function(){
    env <- new.env()
    env$graph <- rdf()
    env$config <- config

    env$config$namespaces <- make_ttl_namespace(env$config, env$graph)
    env$config$add_to_graph <- partial(add_to_graph, .g = env$graph, namespaces = env$config$namespaces)

    env$Programs <- list()
    env$provProgram <- partial(provProgram, config = env$config)
    env$provChannel <- partial(provChannel, config = env$config)
    env$provProgramExecution <- partial(provProgramExecution, config = env$config)
    env$provWasInformedBy <- partial(provWasInformedBy, config = env$config)
    env$provWasPartOf <- partial(provWasPartOf, config = env$config)
    env$provMakeList <- partial(provMakeList, config = env$config)

    env$add_to_namespace <- partial(add_to_namespace, env = env)
    env$save_prov_graph <- partial(save_prov_graph, .graph = env$graph, config = env$config)

    return(env)
}

prov_module <- prov_module_func()