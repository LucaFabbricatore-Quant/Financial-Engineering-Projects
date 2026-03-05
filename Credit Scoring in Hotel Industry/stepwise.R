dataset <- read_excel("C:/Users/clanf/OneDrive/Desktop/progetto gestione del rischio/DATASET_MODELLI_DEFINITIVO.xlsx", 
                      sheet = "Dataset_definitivo")
library(klaR)
dati<- dataset[,4:13]
classi<-as.factor(dataset$Default)
model=NULL;acc=NULL
for (i in 1:100) {
sw_ldmodels<- stepclass(dati, classi, "lda", direction="backward",improvement=0.00001)
sw_ldmodels
model<-c(model,sw_ldmodels$formula);acc<-c(acc,as.numeric(sw_ldmodels$result.pm[1]))
}
PrefModels<-as.data.frame(cbind(model,acc))
PrefModels<-PrefModels[order(acc,decreasing = TRUE),]
for (i in 1:100) {
sw_ldmodels<- stepclass(dati, classi, "lda", direction="both",improvement=0.00001)
sw_ldmodels
model<-c(model,sw_ldmodels$formula);acc<-c(acc,sw_ldmodels$result.pm[1])
}
PrefModels<-as.data.frame(cbind(model,acc))
PrefModels<-PrefModels[order(acc,decreasing = TRUE),]

glmodel<- glm(Default~x1+x2+x3+x4+x5+x6+x7+x8+x9, family=binomial(logit), data=dataset)
summary(glmodel)
library(MASS)
# Selezione stepwise backward (basata su AIC di default)
stepwise_backward <- step(glmodel, direction = "backward")

# Selezione stepwise bidirezionale (both directions)
stepwise_both <- step(glmodel, direction = "both")
