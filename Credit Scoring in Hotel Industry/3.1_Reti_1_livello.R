library(neuralnet)
library(caret)
library(beepr)
library(openxlsx)

library(readxl)
dataset <- read_xlsx("DATASET_MODELLI_DEFINITIVO.xlsx",sheet = "Dataset_definitivo")

set.seed(123)  # Per garantire la riproducibilità
indice_train <- sample(1:nrow(dataset), 0.75 * nrow(dataset))  # Seleziona il 80% dei dati per il training
data_train <- dataset[indice_train, ]  # Crea il dataset di train
data_test <- dataset[-indice_train, ]  # Crea il dataset di test (il restante 20%)

data_trainMOD_STEPWISE <- subset(data_train, select = c(Default,x1,x3,x4,x7,x8,x9))
data_testMOD_STEPWISE <- subset(data_test, select = c(Default,x1,x3,x4,x7,x8,x9))
MeanAccuracy<-NULL
MeanSensitivity<-NULL
MeanSpecificity<-NULL

data_final<-NULL

for(nodes in 1:9) {
  accuracy<-NULL
  sensitivity<-NULL
  specificity<-NULL
  seed<-NULL
  
  while (length(specificity)<100) {
    k<-seq(1:200000000)
    newseed<-sample(k,1)
    set.seed(newseed)
    ANNmodel=neuralnet(Default~x1+x3+x4+x7+x8+x9,data=data_trainMOD_STEPWISE,hidden=nodes,threshold=0.05,act.fct = "logistic",linear.output = FALSE,stepmax=2000000)
    predicted_test<-compute(ANNmodel,data_testMOD_STEPWISE[1:ncol(data_testMOD_STEPWISE)])
    results <- data.frame(actual = data_testMOD_STEPWISE$Default, prediction = predicted_test$net.result)
    roundedresults<-sapply(results,round,digits=0)
    roundedresults<-data.frame(roundedresults)
    roundedresults$actual<-as.factor(roundedresults$actual)
    roundedresults$prediction<-as.factor(roundedresults$prediction)
    CM<-confusionMatrix(roundedresults$prediction,roundedresults$actual,positive="1")  
    newaccuracy<-CM$overall[1]
    newsensitivity<-CM$byClass[1]
    newspecificity<-CM$byClass[2]
    accuracy<-c(accuracy,newaccuracy)
    sensitivity<-c(sensitivity,newsensitivity)
    specificity<-c(specificity,newspecificity)
    seed<-c(seed,newseed)
  }
  
  
  neurons<-rep(nodes,100)
  accuracy<-as.numeric(accuracy)
  sensitivity<-as.numeric(sensitivity)
  specificity<-as.numeric(specificity)
  MeanAccuracy<-c(MeanAccuracy,mean(accuracy))
  MeanSensitivity<-c(MeanSensitivity,mean(sensitivity))
  MeanSpecificity<-c(MeanSpecificity,mean(specificity))
  length(specificity)
  data_nodes<-as.data.frame(cbind(neurons,accuracy,sensitivity,specificity,seed))
  
  write.xlsx(data_nodes, file=paste(paste("C:/Users/enric/OneDrive - Alma Mater Studiorum Universit? di Bologna/work/didattica/corsi/2021/ANNs/results/data_results_", nodes, ".xlsx",sep="")))
  
  data_final<-as.data.frame(rbind(data_final,data_nodes))
  
  assign(paste("data_results", nodes, sep = "_"), data_nodes)
}

write.xlsx(data_final,file=paste("C:/Users/enric/OneDrive - Alma Mater Studiorum Universit? di Bologna/work/ricerca/pubblicazioni/work in progress/A_credit scoring for hotels/results/data_final",".RData",sep=""))

beep()
data_final

CM
