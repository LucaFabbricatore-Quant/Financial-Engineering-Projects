library(readxl)
data_maha <- read_excel("data_maha.xlsx")
s0 <- cov(data_maha[11:20,3:6])
s1 <- cov(data_maha[1:10,3:6])
pooleds <- (s0+s1)/2 #matrice varianza e covarianza media tra le due matrici

#vettore delle medie sia di aziende fallite che sane
mu_0 <- c(mean(data_maha$CR[data_maha$default==0]),c(mean(data_maha$ROA[data_maha$default==0]),c(mean(data_maha$FD[data_maha$default==0]))))
mu_1 <- c(mean(data_maha$CR[data_maha$default==1]),c(mean(data_maha$ROA[data_maha$default==1]),c(mean(data_maha$FD[data_maha$default==1]))))

mdist_0 <- mahalanobis(data_maha[,3:6],mu_0, pooleds)
mdist_1 <- mahalanobis(data_maha[,3:6],mu_1, pooleds)

data_maha$mdist_0 <- mdist_0
data_maha$mdist_1 <- mdist_1
data_maha$prediction <- ifelse(data_maha$mdist_0<data_maha$mdist_1)


mu_1 <- matrix(mu_1,nrow = 1, ncol = 4)
mu_0 <- matrix(mu_0,nrow = 1, ncol = 4)

a <- solve(pooleds)%*%t(mu_1-mu_0)
k <- 0.5*(mu_0%*%solve(pooleds)%*%t(mu_0)-mu_1%*%solve(pooleds)%*%t(mu_1))

data_maha$hx <- as.matrix(data_maha[,3:6])%*%a
data_maha$prediction2 <- ifelse(data_maha$hx>as.numeric(k),1,0)

data_maha$gx <- as.vector(as.matrix(data_maha[3:6])%*%a)-as.numeric(k)

data_maha$prediction3 <- ifelse(data_maha$gx>0,1,0)


