# source("E:\\Data\\condition function_guo\\controlfunctionIV-main\\R")
# 设置文件夹路径
folder_path <- "E:\\testability_IV_JMLR\\control_IV\\controlfunctionIV-main\\R"

# 获取文件夹中所有R脚本的文件名
file_list <- list.files(path = folder_path, pattern = "\\.R$", full.names = TRUE)

# 使用lapply函数逐个调用这些文件
lapply(file_list, source)

# install.packages("readxl")
library(readxl)
# install.packages("Formula") # 如果尚未安装Formula包
library(Formula) # 加载Formula包
# 读取指定工作表
# data <- read_excel("E:\\percentage_dead_patience.xlsx", sheet = "Sheet4")

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
# 计算 MSE
mse <- mean((A - T_A)^2)

# 输出 MSE
print(mse)




