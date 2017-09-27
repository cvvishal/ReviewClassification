install.packages("tsne")
install.packages("Rtsne")
install.packages("rgl")
install.packages("text3D")
library(Rtsne)
library(rgl)
dv<-read.csv("praise.csv",header = FALSE)
nrow(dv)
df_y<-read.csv("ldata.csv")$Praise
df_y<-as.matrix(df_y)
head(df_y)
nrow(df_y)
df_y[which(df_y!=0)]<-1
df_y[which(df_y==0)]<-2
df_y
dfv<-cbind(dv,df_y)
dfv_unique<-unique(dfv)
dfv_matrix<-as.matrix(dfv_unique[1:50])
head(dfv_matrix)
which(duplicated(dfv_matrix))
#ind<-which(duplicated(dfv_matrix))
#dfv_matrix<-dfv_matrix[-ind,]
#nrow(dfv_matrix)
#dfv_unique<-dfv_unique[-ind,]
#nrow(dfv_unique)
tsne_out<-Rtsne(dfv_matrix)
plot(tsne_out$Y,col=as.numeric(dfv_unique$df_y))
plot3d(tsne_out$Y,col=(dfv_unique$df_y),size=10)
dfv_unique$ID <- seq.int(nrow(dfv_unique))

text3d(tsne_out$Y,texts=dfv_unique$ID,col="blue")


