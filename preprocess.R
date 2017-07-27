data<-read.csv("Comment Training Dataset  - Dataset.csv",header = F)
data<-data[-1,]
head(data)
data <- data[-1, ]
head(data)
data<-data[,-c(9,10,11)]
head(data)
data<-data[,-9]
head(data)
data<-data[,-9]
head(data)
data[1100,]
ldata<-data[1:1101,]
ldata[1100,]
udata<-data[-(1:1101),]
udata[1,]
praise_ldata<-ldata[,1:2]
head(praise_ldata)
