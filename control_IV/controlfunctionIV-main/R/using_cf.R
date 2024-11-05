# folder_path <- "E:\\testability_IV_JMLR\\control_IV\\controlfunctionIV-main\\R"
#
# # 获取文件夹中所有R脚本的文件名
# file_list <- list.files(path = folder_path, pattern = "\\.R$", full.names = TRUE)
#
# lapply(file_list, source)
library(readxl)
library(Formula) # 加载Formula包
source("E:\\testability_IV_JMLR\\control_IV\\controlfunctionIV-main\\R\\pretest.R")
source("E:\\testability_IV_JMLR\\control_IV\\controlfunctionIV-main\\R\\cf.R")
using_R_cf <- function(data, Z.id) {

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

# # # # # Example usage:
# setwd("E:\\testability_IV_JMLR\\control_IV\\controlfunctionIV-main\\R")
# data_path <- 'T_5000_0.csv'
# data <- read.table(data_path, header = TRUE, sep = ",")
# Z.id <- "IV2"
# A <- using_R_cf(data, Z.id)
# # print(A)
# T_A <- scale(data[,"A"])
# # 计算 MSE
# mse <- mean((A - T_A)^2)
#
# # 输出 MSE
# print(mse)
