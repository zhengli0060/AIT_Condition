library(readxl)
library(Formula)
# source("pretest.R")
# source("cf.R")
using_R_cf_with_W <- function(data, Z.id) {

  Y <- scale(data[,"Treatment"])
  D <- scale(data[,"Outcome"])
  Z <- as.matrix(data[,c(Z.id)])
  X <- as.matrix(data[,c("W")])
  result.model <- cf(Y~D+I(D^2)+I(D^3)+I(D^4)+X|Z+I(Z^2)+I(Z^3)+X+I(X^2)+I(X^3))
  # summary(cf.model)
  coef <- result.model$coefficients
  A <- Y - coef[2]*D - coef[3]*I(D^2) - coef[4]*I(D^3) - coef[5]*I(D^4) - coef[6]*X

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
