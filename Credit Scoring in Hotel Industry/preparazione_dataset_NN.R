library(openxlsx)

# Carica il dataset
DATASET_PROVA <- read.xlsx("C:/Users/clanf/OneDrive/Desktop/progetto gestione del rischio/dataset_PCA.xlsx",
                           colNames = TRUE,               # Assicurati che la prima riga contenga i nomi delle colonne
                           na.strings = "")               # Gestisce i valori mancanti come NA


# 2. Filtra il dataset per le aziende fallite e sane
fallite <- subset(DATASET_PROVA, Default == 1)
sane <- subset(DATASET_PROVA, Default == 0)

# 3. Dividi il dataset in 80% per il training e 20% per il test
set.seed(123)  # Imposta il seme per la riproducibilità

# Prendi 80% delle aziende sane e 80% delle aziende fallite per il training
sane_train <- sane[sample(1:nrow(sane), size = floor(0.8 * nrow(sane))), ]
fallite_train <- fallite[sample(1:nrow(fallite), size = floor(0.8 * nrow(fallite))), ]

# Combina i dati di addestramento
data_train <- rbind(sane_train, fallite_train)

# Prendi il 20% restante delle aziende sane e fallite per il test
sane_test <- sane[!rownames(sane) %in% rownames(sane_train), ]
fallite_test <- fallite[!rownames(fallite) %in% rownames(fallite_train), ]

# Combina i dati di test
data_test <- rbind(sane_test, fallite_test)

# Controlla la distribuzione nei training e test set
cat("Distribuzione nel dataset di addestramento (data_train):\n")
table(data_train$Default)

cat("\nDistribuzione nel dataset di test (data_test):\n")
table(data_test$Default)
