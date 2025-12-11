

get_files_to_check <- function(root,folder, ignore) {
    #files_fol <- list.files(folder, full.names = TRUE, recursive = TRUE)
    files_root <- list.files(root, full.names = TRUE, recursive = TRUE)
    #files <- c(files_fol, files_root)
    files <- files_root
    files <- files[!basename(files) %in% ignore]
    files <- files[grepl("\\.R$", files)]  # Only keep files ending in .R

    # Exclude program files
    files <- files[!grepl("gen_provone.R", files)]  # Exclude gen_provone.R
    files <- files[!grepl("decorator_func.R", files)]  # Exclude system_prompt_generation.R
    files <- files[!grepl("utils.R", files)]  # Exclude utils
    return(files)
}

entity_marking <- function(entity, config) {
    # Create a marking for the entity based on the config
    if (is.null(config$program$name)) {
        stop("No program name defined in the config")
    }
    prefix <- config$program$name  # Use the first prefix for simplicity
    return(sprintf("%s:%s", prefix, entity))
}

name_concat <- function(...) {
    return(paste(..., sep = "-"))
}

get_unq_id <- function() {
    # Generate a unique identifier
    return(paste0("id_", format(Sys.time(), "%Y%m%d%H%M%S"), "_", sample(1:1000, 1)))
}

get_time_stamp <- function() {
    # Get the current timestamp
    return(format(Sys.time(), "%Y-%m-%dT%H:%M:%S"))
}