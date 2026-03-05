dataset <- read_excel("dataset.xlsx")
library(MASS)
#otteine e visualizza i coefficienti della funzione discriminante
ldmodel=lda(default~x1+x2+x3+x4, data = dataset)
ldmodel
#verifica che le unità siano classificate correttamente dalla funzione 
IntValid <- predict(ldmodel)
IntValid
IntValid$x
IntValid$class

#rappresentare graficamente i risultati
library(caret)
reference <- as.factor(dataset$default)
data <- IntValid$class
confusionMatrix(data,reference,positive="1")

plot(IntValid$x, col="white", xlab="",ylab="", ylim=c(min(IntValid$x)-1,max(IntValid$x)+1))