extractor_env <- new.env()
source("./helpers/provone_llm_functions/extractors/utils.R", local = extractor_env)

find_decorators <- function(file) {
  # Extract the body of the function
  src <- new.env()
  contents <- readLines(file)
  decorators <- extract_decorators(contents)
  return(decorators)
}

extract_decorators <- function(lines) {
  # Match anything like Program(...) or Other(...) after #' @
  pattern <- "(?<=#' @)[A-Za-z0-9_]+\\([^()]*\\)"
  matches <- regmatches(lines, regexpr(pattern, lines, perl = TRUE))
  matches[matches != ""]

  return(matches)
}



decorator_to_triples <- function(decorator, config) {
  # Split the decorator into name and arguments
  extractor_env$process_decorator(decorator, config)
}

process_files <- function(files, config) {
  decorators_list <- list()
  for (file in files) {
    decorators <- find_decorators(file)
    for (decorator in decorators) {
        triples <- decorator_to_triples(decorator, config)
        decorators_list <- c(decorators_list, triples)
    }
  }

  result <- paste(unlist(decorators_list), collapse = "\n")
  return(result)
}