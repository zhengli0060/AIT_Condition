file_list <- list.files(path = folder_path, pattern = "\\.R$", full.names = TRUE)
lapply(file_list, source)
library(readxl)
library(Formula)


# data("nonlineardata")
# Y <- log(nonlineardata[,"insulin"])
# D <- nonlineardata[,"bmi"]
# Z <- as.matrix(nonlineardata[,c("Z.1","Z.2","Z.3","Z.4")])
# X <- as.matrix(nonlineardata[,c("age","sex")])
# cf.model <- cf(Y~D+I(D^2)+X|Z+I(Z^2)+X)
# summary(cf.model)

Y <- scale(data[,"Treatment"])
D <- scale(data[,"Outcome"])
Z <- as.matrix(data[,c("IV2")])
X <- as.matrix(data[,c("W")])
cf.model <- cf(Y~D+I(D^2)+X|Z+I(Z^2)+X)
summary(cf.model)
coef <- cf.model$coefficients
A <- Y - coef[2]*D - coef[3]*I(D^2) - coef[4]*X
T_A <- scale(data[,"A"])
mse <- mean((A - T_A)^2)
print(mse)




