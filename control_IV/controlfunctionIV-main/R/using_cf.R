library(readxl)
library(Formula)


extract_columns <- function(data, pattern = "^W") {
  # Find column names starting with the specified pattern
  cols <- grep(pattern, colnames(data), value = TRUE)

  if (length(cols) == 0) {
    message("No columns starting with the pattern '", pattern, "' found in the dataset.")
    return(NULL)
  }

  # Extract matching columns and convert to a matrix
  as.matrix(data[, cols, drop = FALSE])
}

using_R_cf_with_W <- function(data, Z.id) {

  Y <- scale(data[,"Treatment"])
  D <- scale(data[,"Outcome"])
  Z <- as.matrix(data[,c(Z.id)])
  # X <- as.matrix(data[,c("W1")])
  # X2 <- as.matrix(data[,c("W2")])
  X <- extract_columns(data, pattern = "^W")
  result.model <- cf(Y~D+I(D^2)+I(D^3)+I(D^4)+ X|Z+I(Z^2)+I(Z^3)+X+I(X^2)+I(X^3))

  # summary(cf.model)
  coef <- result.model$coefficients
  A <- Y - coef[2]*D - coef[3]*I(D^2) - coef[4]*I(D^3) - coef[5]*I(D^4) - X %*% coef[6:(6 + ncol(X) - 1)]
}



using_R_cf_no_W <- function(data, Z.id) {

  Y <- scale(data[,"Treatment"])
  D <- scale(data[,"Outcome"])
  Z <- as.matrix(data[,c(Z.id)])
  result.model <- cf(Y~D+I(D^2)+I(D^3)+I(D^4)|Z+I(Z^2)+I(Z^3))
  # summary(cf.model)
  coef <- result.model$coefficients
  A <- Y - coef[2]*D - coef[3]*I(D^2) - coef[4]*I(D^3) - coef[5]*I(D^4)

}

# # # # # Example usage:
# source("pretest.R")
# source("cf.R")
# data_path <- 'D:\\pythonProject\\AIT_Condition\\example_data\\Example_Data_compare_PIM_5000.csv'
# data <- read.table(data_path, header = TRUE, sep = ",")
# Z.id <- "IV1"
# A <- using_R_cf_with_W(data, Z.id)

