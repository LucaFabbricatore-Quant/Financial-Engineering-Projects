library(readxl)
dataset <- read_excel("dataset.xlsx")


dataset$x1mod <- dataset$x1*100
dataset$x2mod <- dataset$x2*100
dataset$x4mod <- dataset$x4*100
boxplot(dataset$x1-dataset$status)
boxplot(dataset$x2-dataset$status.outline=F)
boxplot(dataset$x3-dataset$status.outline=F)
boxplot(dataset$x4-dataset$status.outline=F)


lmodel <- lm(dataset$default~dataset$x1mod+dataset$x2mod+dataset$x3+dataset$x4mod)
summary(lmodel)
