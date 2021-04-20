#load library
library(ruimtehol)
library(itertools2)
require(magrittr) 
require(caret)
######################
set.seed(177)

kmer <- function(x,k=5){
  b=nchar(x[1])
  for(i in 1:length(x)){
    delta1=paste(substr(x[i],1,100),substr(x[i],107,206),sep = '')
    delta2=NULL
    for(j in 1:(b-6-k-1)){
      delta2 <- c(delta2,substr(delta1,j,j+k-1))
    }
    x[i] <- paste(delta2,sep = '',collapse = ' ')
  }
  return(x)
}

fileNames <- dir('.')[c(1:11,24)]
#fileNames='AATAAA.txt'
total_access={}
for(filename in fileNames){
  pData <- read.csv(filename,header = F,stringsAsFactors = F)[,1]  %>% as.vector() %>% kmer()
  negData <- read.csv(paste('neg',filename,sep = ''),header = F,stringsAsFactors = F)[,1] %>% as.vector() %>% kmer()
  data = data.frame(rbind(cbind(pData,label=1), cbind(negData,label=0 )),stringsAsFactors = F)
  
  require(caret)
  folds<-createFolds(y=data$label,k=10) #根据training的laber-Species把数据集切分成10等份
  
  access={}
  for(i in 1:10){
    train_x <- data[-folds[[i]],1]
    train_y <- data[-folds[[i]],2]
  
    test_x  <- data[folds[[ i]],1]
    test_y  <- data[folds[[ i]],2]
    #建立starspace模型
    model <- embed_tagspace(x=train_x,y=train_y,
                            dim = 30,epoch = 1, loss = "hinge", adagrad = T, 
                            similarity = "dot", negSearchLimit = 10,ngrams = 10,
                            minCount = 5)
    
    #结果评估
    result <- predict(model,test_x)
    TN=TP=FN=FP=0
    for(j in 1:length(test_x)){
      if(test_y[j]==1 & result[[j]]$prediction[1,1]==1){TP=TP+1 }
      if(test_y[j]==1 & result[[j]]$prediction[1,1]==0){FP=FP+1 }
      if(test_y[j]==0 & result[[j]]$prediction[1,1]==0){TN=TN+1 }
      if(test_y[j]==0 & result[[j]]$prediction[1,1]==1){FN=FN+1 }
    }
    Sn = TP/(TP+FN)
    Sp=TN/(TN+FP)
    Accuracy = (TP+TN)/(TN+FP+TP+FN)
    MCC=(TP*TN-FP*FN)/sqrt((TP+FP)*(TN+FN)*(TN+FP)*(TP+FN))
    AUC=(Sn+Sp)/2
    Fscore=(2*TP)/(2*TP+FP+FN)
    access=rbind(access,data.frame(TP,TN,FP,FN,Sn,Sp,Accuracy,MCC,AUC,Fscore))
  }
  total_access=rbind(total_access,apply(access,2,mean))
}
rownames(total_access) <- sapply(fileNames,function(x) substr(x,1,6),USE.NAMES = F)
total_access


