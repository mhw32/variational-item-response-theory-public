#!/usr/bin/env Rscript

# NOTE: EM cannot be used to infer ability. It is only a model 
# for missing data prediction.

library("mirt")
library("optparse")
library("reticulate")
library("jsonlite")

option_list = list(
    make_option(c("-d", "--dataset"), type="character", default="./response.npy", help="Path to response.npy file"),
    make_option(c("-i", "--irt"), type="character", default="2PL", help="Type of IRT model"),
    make_option(c("-a", "--abilitydim"), type="integer", default=1, help="Number of ability dimensions"),
    make_option(c("-o", "--itemfile"), type="character", default="./irt_em_inferred_item_features.json", help="filepath for where to save json file"),
    make_option(c("-m", "--missingfile"), type="character", default="./irt_em_missing_data_imputation.json", help="filepath for where to save numpy array")
)

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

np <- import("numpy")
mat <- np$load(opt$dataset)

# replace -1 with NA to indicate missing data
mat <- ifelse(mat == -1, NA, mat)

# MIRT requires named columns
mat <- data.frame(mat)

ptm <- proc.time()  # start clock

model <- mirt::mirt(
    mat, 
    opt$abilitydim, 
    verbose=TRUE, 
    method = "EM", 
    guess = 0, 
    upper = 1, 
    quadpts = 61, 
    draws = 5000, 
    itemtype = opt$irt,
    removeEmptyRows = TRUE
)
results <- coef(model)
results <- data.matrix(results)

# user time relates to the execution of the code
# system time relates to system processes such as opening and closing files
# elapsed time is the difference in times since you started the stopwatch
proc.time() - ptm  # stop clock

write_json(results, opt$itemfile)

# fill in missing data
scores <- fscores(model, method = 'MAP')
mat_full <- imputeMissing(model, scores)

write_json(mat_full, opt$missingfile)
