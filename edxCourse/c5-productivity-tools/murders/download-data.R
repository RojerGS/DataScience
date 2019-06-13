url <- "https://raw.githubusercontent.com/rafalab/dslabs/master/inst/extdata/murders.csv"
dest_file <- "rawdata/murders.csv"
download.file(url, destfile=dest_file)