library(FactoMineR)
library(Rcmdr)

DATASET_PROVA <- 
  readXL("C:/Users/clanf/OneDrive/Desktop/progetto gestione del rischio/DATASET_MODELLI_DEFINITIVO.xlsx",
         rownames=FALSE, header=TRUE, na="", sheet="Dataset_definitivo", 
         stringsAsFactors=TRUE)
DATASET_PROVA.PCA<-DATASET_PROVA[, c("x1", "x2", "x3", "x4", "x5", "x6", 
                                     "x7", "x8", "x9", "x10")]
PCA_prova<-PCA(DATASET_PROVA.PCA , scale.unit=TRUE, ncp=10, graph = FALSE)
PCA_prova.hcpc<-HCPC(PCA_prova ,nb.clust=-1,consol=FALSE,min=3,max=10,
                     graph=TRUE)
PCA_prova
print(plot.PCA(PCA_prova, axes=c(1, 2), choix="ind", habillage="none", 
               col.ind="black", col.ind.sup="blue", col.quali="magenta", label=c("ind", 
                                                                                 "ind.sup", "quali"),new.plot=TRUE))
print(plot.PCA(PCA_prova, axes=c(1, 2), choix="var", new.plot=TRUE, 
               col.var="black", col.quanti.sup="blue", label=c("var", "quanti.sup"), 
               lim.cos2.var=0))
summary(PCA_prova, nb.dec = 3, nbelements=10, nbind = 10, ncp = 3, file="")
PCA_prova$eig
PCA_prova$var
PCA_prova$ind
remove(DATASET_PROVA.PCA)
editDataset(DATASET_PROVA)

PCA_prova$var$cor

#creazione dataset per i modelli

# Estrarre le coordinate delle osservazioni sulle componenti principali
dataset_pca <- as.data.frame(PCA_prova$ind$coord)
dataset_pca

dataset_PCA_MODELLI <- dataset_pca[,c("Dim.1","Dim.2","Dim.3","Dim.4","Dim.5")]
dataset_PCA_MODELLI
write.csv(dataset_PCA_MODELLI, "dataset_PCA_MODELLI.csv", row.names = FALSE)




