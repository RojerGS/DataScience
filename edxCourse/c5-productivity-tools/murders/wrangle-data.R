library(tidyverse)

murders <- read_csv("rawdata/murders.csv")
murders <- murders %>% mutate(region=factor(region), rate=total/population*10^5)
save(murders, file="rdata/murders.rda")