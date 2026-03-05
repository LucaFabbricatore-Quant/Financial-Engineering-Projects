library(MASS)
# ottiene e visualizza i coefficienti della funzione discriminante
library(MASS)
library(readxl)  # Per leggere file Excel (.xlsx)

library(readxl)

dataset <- read_excel("dataset_PCA.xlsx")

# Assicurati che Default sia un fattore
dataset$Default <- as.factor(dataset$Default)

library(MASS)
ldmodel <- lda(Default ~ x1 + x2 + x3 + x4 + x5, data = dataset)



ldmodel=lda(Default~x1+x2+x3+x4 +x5,data=dataset)
ldpred<-predict(ldmodel)
dataset$x1mod=dataset$x1*100
dataset$x2mod=dataset$x2*100
dataset$x4mod=dataset$x4*100
glmodel<- glm(Default ~ x1 + x2 + x3 + x4 + x5, family=binomial(logit), data=dataset)
glmpred<-predict(glmodel, type = "response")
# disegna la curva ROC per il modello di analisi discriminante lineare
library(ROCR)
data<-ldpred$posterior[,2]
reference<-as.factor(dataset$Default)
ROCRpred_ld<-prediction(data, reference)
ROCRperf_ld<-performance(ROCRpred_ld,measure = "tpr", x.measure = "fpr")
plot(ROCRperf_ld)
abline(0,1,lty=2)
# disegna la curva ROC per il modello di regressione logistica
data<-glmpred
reference<-as.factor(dataset$Default)
ROCRpred_gl <- prediction(glmpred , dataset$Default)
ROCRperf_gl <- performance(ROCRpred_gl,measure = "tpr", x.measure = "fpr")
plot(ROCRperf_gl,col="red",add=TRUE)
abline(0,1,lty=2)

AUC_ld<-performance(ROCRpred_ld,"auc")@y.values[[1]]
AUC_gl<-performance(ROCRpred_gl,"auc")@y.values[[1]]
# incremento percentuale di AUC ottenuto passando dal modello di analisi discriminante alla regressione logistica
deltaperc_AUC<-(AUC_gl-AUC_ld)/AUC_ld
deltaperc_AUC
gini_ld<-2*AUC_ld-1
gini_ld
gini_gl<-2*AUC_gl-1
gini_gl
# incremento percentuale di gini ottenuto passando dal modello di analisi discriminante alla regressione logistica
deltaperc_gini<-(gini_gl-gini_ld)/gini_ld
deltaperc_gini

