data<-read.csv("Comment Training Dataset  - Dataset.csv",header = F)
data<-data[-1,]
head(data)
#data <- data[-1, ]
head(data)
data<-data[,-c(9,10,11)]
head(data)
data<-data[,-9]
head(data)
data<-data[,-9]
names(data) <- as.matrix(data[1, ])
data <- data[-1, ]
data[] <- lapply(data, function(x) type.convert(as.character(x)))
head(data)
data[1100,]
ldata<-data[1:1101,]
ldata[1100,]
ldata<-ldata[complete.cases(ldata),]
ldata<-ldata[!ldata$Comments=="",]
write.csv(ldata,"lldata.csv",row.names = FALSE)
udata<-data[-(1:1101),]
udata[1,]
praise_ldata<-ldata[,1:2]
head(praise_ldata)
