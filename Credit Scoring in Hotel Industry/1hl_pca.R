library(neuralnet)
library(caret)
library(beepr)
library(openxlsx)

data_trainMOD <- subset(data_train, select = c(Default,x1,x2,x3,x4,x5))
data_testMOD <- subset(data_test, select = c(Default,x1,x2,x3,x4,x5))
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
    ANNmodel=neuralnet(Default~x1+x2+x3+x4+x5,data=data_trainMOD,hidden=nodes,threshold=0.05,act.fct = "logistic",linear.output = FALSE,stepmax=2000000)
    predicted_test<-compute(ANNmodel,data_testMOD[1:ncol(data_testMOD)])
    results <- data.frame(actual = data_testMOD$Default, prediction = predicted_test$net.result)
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
  
  #write.xlsx(data_nodes, file=paste(paste("C:/Users/enric/OneDrive - Alma Mater Studiorum Universit? di Bologna/work/didattica/corsi/2021/ANNs/results/data_results_", nodes, ".xlsx",sep="")))
  
  data_final<-as.data.frame(rbind(data_final,data_nodes))
  
  assign(paste("data_results", nodes, sep = "_"), data_nodes)
}

write.xlsx(data_final,file=paste("C:/Users/enric/OneDrive - Alma Mater Studiorum Universit? di Bologna/work/ricerca/pubblicazioni/work in progress/A_credit scoring for hotels/results/data_final",".RData",sep=""))

beep()

