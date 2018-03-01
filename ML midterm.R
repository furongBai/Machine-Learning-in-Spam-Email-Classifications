#########################################################################
######################## Input Data Set Into R ##########################
#########################################################################

# Uncommet to install packages
# install.packages("corrplot")
# install.packages("kernlab")
# if using Mac, uncomment following line to install package
# install.packages("doMC") 
# if using Microsoft, uncommet following line to install package
# install.packages("doMC", repos="http://R-Forge.R-project.org")
library(corrplot)
library(caret)
library(dplyr)
library(ggplot2)
require(kernlab)
require(doMC)

# input the data set
spamD <- read.table('http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data',sep=',',header=F)
colnames(spamD) <- c(
  'word.freq.make', 'word.freq.address', 'word.freq.all',
  'word.freq.3d', 'word.freq.our', 'word.freq.over', 'word.freq.remove',
  'word.freq.internet', 'word.freq.order', 'word.freq.mail',
  'word.freq.receive', 'word.freq.will', 'word.freq.people',
  'word.freq.report', 'word.freq.addresses', 'word.freq.free',
  'word.freq.business', 'word.freq.email', 'word.freq.you',
  'word.freq.credit', 'word.freq.your', 'word.freq.font',
  'word.freq.000', 'word.freq.money', 'word.freq.hp', 'word.freq.hpl',
  'word.freq.george', 'word.freq.650', 'word.freq.lab',
  'word.freq.labs', 'word.freq.telnet', 'word.freq.857',
  'word.freq.data', 'word.freq.415', 'word.freq.85',
  'word.freq.technology', 'word.freq.1999', 'word.freq.parts',
  'word.freq.pm', 'word.freq.direct', 'word.freq.cs',
  'word.freq.meeting', 'word.freq.original', 'word.freq.project',
  'word.freq.re', 'word.freq.edu', 'word.freq.table',
  'word.freq.conference', 'char.freq.semi', 'char.freq.lparen',
  'char.freq.lbrack', 'char.freq.bang', 'char.freq.dollar',
  'char.freq.hash', 'capital.run.length.average',
  'capital.run.length.longest', 'capital.run.length.total',
  'spam'
)
View(spamD)
str(spamD)


#########################################################################
############################# Data Cleaning #############################
#########################################################################

#Check for missing values
sapply(spamD, function(x) sum(is.na(x)))

#Check the class of each var
sapply(spamD, function(x) class(x))

#Create my.summary function for numerical variables
my.summary <- function(input_df){
  summary_df <- 
    data.frame(quantile_.01 = sapply(input_df, function(x) quantile(x,.01)),
               quantile_.05 = sapply(input_df, function(x) quantile(x,.05)),
               quantile_.1 = sapply(input_df, function(x) quantile(x,.1)),
               quantile_.25 = sapply(input_df, function(x) quantile(x,.25)),
               quantile_.5 = sapply(input_df, function(x) quantile(x,.5)),
               quantile_.75 = sapply(input_df, function(x) quantile(x,.75)),
               quantile_.95 = sapply(input_df, function(x) quantile(x,.95)),
               quantile_.99 = sapply(input_df, function(x) quantile(x,.99)),
               mean = sapply(input_df, mean),
               variance = sapply(input_df, var),
               min = sapply(input_df, min),
               max = sapply(input_df, max),
               percMissing = sapply(input_df, function(x) sum(is.na(x)))/nrow(input_df)
    )
}

#########################################################################
########################### Data Exploration ############################
#########################################################################
#find all numeric variables
isnumeric <-sapply(spamD, function(x) is.numeric(x))

#run my.summary on all of the numeric varaibles
summary_numeric <- my.summary(spamD[,isnumeric])

write.csv(summary_numeric, file="summary_numeric.csv", row.names=TRUE, quote = FALSE) 


#create correlation plot
cor_spamD<-cor(spamD[,isnumeric])
corrplot(cor_spamD, method="circle")

#Boxplot for all numeric variables against spam_bool
par(mfrow=c(2, 7))

sapply(seq_along(spamD[,isnumeric]), function(i) {
  x <- spamD[,isnumeric][, i]
  boxplot(x ~ spamD$spam, 
          main=names(spamD[isnumeric])[i] 
  )})


#########################################################################
########################## Data Prep (Question 2) #######################
#########################################################################


###### Splitting to training & testing ########
set.seed(20)
dt = sort(sample(nrow(spamD), nrow(spamD)*0.8))
training1 <- spamD[dt,]
testing1 <- spamD[-dt,]

########### Standardization#################
values <- preProcess(training1[,1:57], method = c("center","scale"))
training2 <-predict(values, training1[ ,1:57])
training <- mutate(training2, spam=training1[,58])

testing2 <- predict(values, testing1[ ,1:57])
testing <- mutate(testing2, spam=testing1[,58])



#########################################################################
############################## Adaline ##################################
#########################################################################
adalineGD <- function(X, y, n.iter, eta) {
  
  # extend input vector and initialize extended weight
  X[, dim(X)[2] + 1] <- 1 
  X <- as.matrix(X)
  w <- as.matrix(rep(0, dim(X)[2]))
  
  # initialize cost values - gets updated according to epochnums - number of epochs
  cost <- rep(0, n.iter)
  errors <- rep(0, n.iter)
  TN = rep(0, n.iter)
  FN = rep(0, n.iter)
  FP = rep(0, n.iter)
  TP = rep(0, n.iter)
  Accuracy <- rep(0, n.iter)
  Error <- rep(0, n.iter)
  Precision <- rep(0, n.iter)
  Recall <- rep(0, n.iter)
  FPR <- rep(0, n.iter)
  
  # loop over the number of epochs
  for (i in 1:n.iter) {
    
    # find the number of wrong prediction before weight update
    for (j in 1:dim(X)[1]) {
      
      # compute net input
      z <- sum(w * X[j, ])
      
      # quantizer
      if (z < 0) {
        ypred <- -1
      }else {
        ypred <- 1
      }
      
      # comparison with actual values and counting error
      if(ypred != y[j]) {
        errors[i] <- errors[i] + 1
      }
      
      # metrices
      if(ypred==-1 && y[j]==-1){TN[i] = TN[i]+1}
      if(ypred==-1 && y[j]== 1){FN[i] = FN[i]+1}
      if(ypred== 1 && y[j]==-1){FP[i] = FP[i]+1}
      if(ypred== 1 && y[j]==1){TP[i] = TP[i]+1}
      
      Accuracy[i] <- (TP[i]+TN[i])/(FP[i]+FN[i]+TP[i]+TN[i])
      Error[i] <- 1-Accuracy[i]
      Precision[i] <- TP[i]/(TP[i]+FP[i])
      Recall[i] <- TP[i]/(TP[i]+FN[i])
      FPR[i] <- FP[i]/(FP[i]+TN[i])
    }
    
    
    # update cost function (SSE)
    cost[i] <- sum((y - X %*% w)^2)/2
    
    # update weight according to gradient descent
    p = t(X) %*% (y - X %*% w)
    w <- w + eta* p
  }
  
  # data frame consisting of cost and error info
  confusion_matrix <- matrix(c(TN[n.iter],FN[n.iter],FP[n.iter],TP[n.iter]),nrow = 2)
  colnames(confusion_matrix) <- c("Negative","Positive")
  rownames(confusion_matrix) <- c("Negative","Positive")
  
  infomatrix <- matrix(rep(0, 8 * n.iter), nrow = n.iter, ncol = 8)
  infomatrix[, 1] <- 1:n.iter
  infomatrix[, 2] <- log(cost)
  infomatrix[, 3] <- errors
  infomatrix[, 4] <- Accuracy
  infomatrix[, 5] <- Error
  infomatrix[, 6] <- Precision
  infomatrix[, 7] <- Recall
  infomatrix[, 8] <- FPR
  
  infodf <- as.data.frame(infomatrix)
  names(infodf) <- c("No_iteration", "cost_function", "errors","Accuracy","Error","Precision","Recall","FPR")
  
  infolist <- list(w,infodf,confusion_matrix)
  names(infolist) <- c("w","infomatrix","confusion_matrix")
  
  return(infolist)
}

predict.adaline <- function(w, X, y){
  # extend input vector and initialize extended weight
  X[, dim(X)[2] + 1] <- 1 
  X <- as.matrix(X)
  
  
  # initialize metrics
  errors <- 0
  TN = 0
  FN = 0
  FP = 0
  TP = 0
  Accuracy <- 0
  Error <- 0
  Precision <- 0
  Recall <- 0
  FPR <- 0
  
  # find the number of wrong prediction
  for (j in 1:dim(X)[1]) {
    
    # compute net input
    z <- sum(w * X[j, ])
    
    # quantizer
    if (z < 0) {
      ypred <- -1
    }else {
      ypred <- 1
    }
    
    # comparison with actual values and counting error
    if(ypred != y[j]) {
      errors <- errors + 1
    }
    
    # metrices
    if(ypred==-1 && y[j]==-1){TN = TN+1}
    if(ypred==-1 && y[j]== 1){FN = FN+1}
    if(ypred== 1 && y[j]==-1){FP = FP+1}
    if(ypred== 1 && y[j]==1){TP = TP+1}
    
    Accuracy <- (TP+TN)/(FP+FN+TP+TN)
    Error <- 1-Accuracy
    Precision <- TP/(TP+FP)
    Recall <- TP/(TP+FN)
    FPR <- FP/(FP+TN)
  }
  
  # data frame consisting of cost and error info
  confusion_matrix <- matrix(c(TN,FN,FP,TP),nrow = 2)
  colnames(confusion_matrix) <- c("Negative","Positive")
  rownames(confusion_matrix) <- c("Negative","Positive")
  
  infomatrix <- matrix(rep(0, 7), nrow = 1, ncol = 7)
  infomatrix[, 1] <- "Matrics"
  infomatrix[, 2] <- errors
  infomatrix[, 3] <- Accuracy
  infomatrix[, 4] <- Error
  infomatrix[, 5] <- Precision
  infomatrix[, 6] <- Recall
  infomatrix[, 7] <- FPR
  
  infodf <- as.data.frame(infomatrix)
  names(infodf) <- c("Matrics", "errors","Accuracy","Error","Precision","Recall","FPR")
  
  infolist <- list(infodf,confusion_matrix)
  names(infolist) <- c("infomatrix","confusion_matrix")
  
  return(infolist)
  
}

########## Apply to the training set. #############
y1 <- rep(1, nrow(training))
y1[training[,58]==0] <- -1
result.adalineGD <- adalineGD(training[,-58], y1 ,n.iter = 100, eta = 0.00001)
# check the weights
result.adalineGD$w
# check confusion matrix
confusion_matrix = result.adalineGD$confusion_matrix
confusion_matrix
# check error and other matrics in each iteration
View(result.adalineGD$infomatrix)

###### Model Evaluation ###########
# plot cost function minimization process
ggplot(result.adalineGD$infomatrix, aes(x = No_iteration, y = cost_function)) + 
  geom_line(size = 0.8, col = "pink") +
  xlab("No. of iteration") + 
  ylab("log(SSE)") +
  ggtitle("Adaline - Minimizing Cost Function with GD: eta = 0.00001")

# plot accuracy as a function of learning effort
ggplot(result.adalineGD$infomatrix, aes(x = No_iteration, y = Accuracy)) +
  geom_line() +
  ggtitle("Learning curve")

# plot ROC (Receiver Operating Characteristic) curve 
c = rep(0,8)
ggplot(rbind(c,result.adalineGD$infomatrix), aes(x = FPR, y = Recall)) + 
  geom_line(col="blue",size=1)+ 
  geom_line(aes(y=FPR),linetype=2,size=0.6 ) +
  ggtitle("ROC curve (Receiver Operating Characteristic)")

# plot precision vs. recall 
baseline = rep(((confusion_matrix[1,2]+confusion_matrix[2,2])/nrow(training)),nrow(result.adalineGD$infomatrix))
ggplot(cbind(result.adalineGD$infomatrix, baseline), aes(x = Precision, y = Recall)) + 
  geom_line(col="red",size=1) +
  geom_line(aes(y=baseline),linetype=2,size=0.6) +
  ylim(0, 1) +
  ggtitle("Precision-recall curve")


# Plot cost function using various learning rate
result1 <- adalineGD(training[,-58], y1 ,n.iter = 100, eta = 0.00001)
result2 <- adalineGD(training[,-58], y1 ,n.iter = 100, eta = 0.000005)
result1 <- result1$infomatrix
result2 <- result2$infomatrix
label <- rep("0.00001", dim(result1)[1])
result1 <- cbind(label, result1)
label <- rep("0.000005", dim(result2)[1])
result2 <- cbind(label, result2)
df <- rbind(result1, result2)
ggplot(df, aes(x = No_iteration, y = cost_function)) + 
  geom_line(aes(color=label, linetype=label), size = 1) +
  xlab("No. of iteration") + 
  ylab("log(SSE)") +
  ggtitle("Adaline GD - Cost function with various learning rates")


######## Test model on testing set #########
y2 <- rep(1, nrow(testing))
y2[testing[,58]==0] <- -1
w = result.adalineGD$w
test.adalineGD = predict.adaline(w, testing[,-58], y2)
test.adalineGD








#########################################################################
############################ Logistic Regression ########################
#########################################################################

logisticGD <- function(X, y, n.iter, eta) {
  
  # extend input vector and initialize extended weight
  X[, dim(X)[2] + 1] <- 1 
  X <- as.matrix(X)
  w <- as.matrix(rep(0, dim(X)[2]))
  
  # initialize cost values - gets updated according to epochnums - number of epochs
  cost <- rep(0, n.iter)
  errors <- rep(0, n.iter)
  
  # initialize various metrics
  TN = rep(0, n.iter)
  FN = rep(0, n.iter)
  FP = rep(0, n.iter)
  TP = rep(0, n.iter)
  Accuracy <- rep(0, n.iter)
  Error <- rep(0, n.iter)
  Precision <- rep(0, n.iter)
  Recall <- rep(0, n.iter)
  FPR <- rep(0, n.iter)
  
  #Sigmoid function
  sigmoid <- function(z)
  {
    g <- 1/(1+exp(-z))
    return(g)
  }
  
  # loop over the number of epochs
  for (i in 1:n.iter) {
    
    # find the number of wrong prediction before weight update
    for (j in 1:dim(X)[1]) {
      
      # compute net input
      z <- sigmoid(sum(w * X[j, ]))
      #z <- 1/(1+exp(-sum(w * X[j, ])))
      
      # quantizer
      if (z < 0.5) {
        ypred <- 0
        }else {
        ypred <- 1
        }
      
      # comparison with actual labels and counting error
      if(ypred != y[j]) {
        errors[i] <- errors[i] + 1
      }
      
      # metrics
      if(ypred==0 && y[j]==0){TN[i] = TN[i]+1}
      if(ypred==0 && y[j]== 1){FN[i] = FN[i]+1}
      if(ypred== 1 && y[j]==0){FP[i] = FP[i]+1}
      if(ypred== 1 && y[j]==1){TP[i] = TP[i]+1}
      
      Accuracy[i] <- (TP[i]+TN[i])/(FP[i]+FN[i]+TP[i]+TN[i])
      Error[i] <- 1-Accuracy[i]
      Precision[i] <- TP[i]/(TP[i]+FP[i])
      Recall[i] <- TP[i]/(TP[i]+FN[i])
      FPR[i] <- FP[i]/(FP[i]+TN[i])
    }
    
    # update the cost function(cost function is the formula given in classnotes)
    # cost[i] <- sum((y - X %*% w)^2)/2
    g <- sigmoid(X %*% w)
    cost[i] <- sum((-y*log(g)) - ((1-y)*log(1-g)))
    
    # update weight according to gradient descent
    p = t(X) %*% (y - g)
    w <- w + eta* p
  }
  
  # data frame consisting of cost and error info
  confusion_matrix <- matrix(c(TN[n.iter],FN[n.iter],FP[n.iter],TP[n.iter]),nrow = 2)
  colnames(confusion_matrix) <- c("Negative","Positive")
  rownames(confusion_matrix) <- c("Negative","Positive")
  
  infomatrix <- matrix(rep(0, 8 * n.iter), nrow = n.iter, ncol = 8)
  infomatrix[, 1] <- 1:n.iter
  infomatrix[, 2] <- log(cost)
  infomatrix[, 3] <- errors
  infomatrix[, 4] <- Accuracy
  infomatrix[, 5] <- Error
  infomatrix[, 6] <- Precision
  infomatrix[, 7] <- Recall
  infomatrix[, 8] <- FPR
  
  infodf <- as.data.frame(infomatrix)
  names(infodf) <- c("No_iteration", "cost_function", "errors","Accuracy","Error","Precision","Recall","FPR")
  
  infolist <- list(w,infodf,confusion_matrix)
  names(infolist) <- c("w","infomatrix","confusion_matrix")
  
  return(infolist)
}

predict.logistic <- function(w, X, y){
  # extend input vector and initialize extended weight
  X[, dim(X)[2] + 1] <- 1 
  X <- as.matrix(X)
  
  # initialize metrics
  errors <- 0
  TN = 0
  FN = 0
  FP = 0
  TP = 0
  Accuracy <- 0
  Error <- 0
  Precision <- 0
  Recall <- 0
  FPR <- 0
  
  #Sigmoid function
  sigmoid <- function(z)
  {
    g <- 1/(1+exp(-z))
    return(g)
  }
  
  # find the number of wrong prediction
  for (j in 1:dim(X)[1]) {
    
    # compute net input
    z <- sigmoid(sum(w * X[j,]))
    
    # quantizer
    if (z < 0.5) {
      ypred <- 0
    }else {
      ypred <- 1
    }
    
    # comparison with actual values and counting error
    if(ypred != y[j]) {
      errors <- errors + 1
    }
    
    # metrics
    if(ypred==0 && y[j]==0){TN = TN+1}
    if(ypred==0 && y[j]== 1){FN = FN+1}
    if(ypred== 1 && y[j]==0){FP = FP+1}
    if(ypred== 1 && y[j]==1){TP = TP+1}
    
    Accuracy <- (TP+TN)/(FP+FN+TP+TN)
    Error <- 1-Accuracy
    Precision <- TP/(TP+FP)
    Recall <- TP/(TP+FN)
    FPR <- FP/(FP+TN)
  }
  
  # data frame consisting of cost and error info
  confusion_matrix <- matrix(c(TN,FN,FP,TP),nrow = 2)
  colnames(confusion_matrix) <- c("Negative","Positive")
  rownames(confusion_matrix) <- c("Negative","Positive")
  
  infomatrix <- matrix(rep(0, 7), nrow = 1, ncol = 7)
  infomatrix[, 1] <- "Matrics"
  infomatrix[, 2] <- errors
  infomatrix[, 3] <- Accuracy
  infomatrix[, 4] <- Error
  infomatrix[, 5] <- Precision
  infomatrix[, 6] <- Recall
  infomatrix[, 7] <- FPR
  
  infodf <- as.data.frame(infomatrix)
  names(infodf) <- c("Matrics", "errors","Accuracy","Error","Precision","Recall","FPR")
  
  infolist <- list(infodf,confusion_matrix)
  names(infolist) <- c("infomatrix","confusion_matrix")
  
  return(infolist)
  
}

####### Apply to the training set. #########
y1=training[,58]
result.logGD <- logisticGD(training[,-58], y1 ,n.iter = 100, eta = 0.00015)
# check weights
result.logGD$w
# check confusion matrix
confusion_matrix = result.logGD$confusion_matrix
confusion_matrix
# check error and cost function in each iteration
View(result.logGD$infomatrix)


###### Model Evaluation ###########

# plot cost function minimization process
ggplot(result.logGD$infomatrix, aes(x = No_iteration, y = cost_function)) + 
  geom_line(size = 0.8, col = "pink") +
  xlab("No. of iteration") + 
  ylab("Cost") +
  ggtitle("Logistic - Minimizing Cost Function with GD: eta = 0.00015")

# plot accuracy as a function of learning effort
ggplot(result.logGD$infomatrix, aes(x = No_iteration, y = Accuracy)) + 
  geom_line(size = 0.8, col = "rosybrown2") + 
  ggtitle("Learning curve")

# plot ROC (Receiver Operating Characteristic) curve 
c = rep(0,8)
ggplot(rbind(c,result.logGD$infomatrix), aes(x = FPR, y = Recall)) + 
  geom_line(col="blue",size=1)+ 
  geom_line(aes(y=FPR),linetype=2,size=0.6 ) +
  ggtitle("ROC curve (Receiver Operating Characteristic)")
  
# plot precision vs. recall 
baseline = rep(((confusion_matrix[1,2]+confusion_matrix[2,2])/nrow(training)),nrow(result.logGD$infomatrix))
ggplot(cbind(result.logGD$infomatrix, baseline), aes(x = Precision, y = Recall)) + 
  geom_line(col="red",size=1) +
  geom_line(aes(y=baseline),linetype=2,size=0.6) +
  ylim(0, 1) +
  ggtitle("Precision-recall curve")


# Plot cost function using various learning rate
result1 <- logisticGD(training[,-58], y1 ,n.iter = 100, eta = 0.00005)
result2 <- logisticGD(training[,-58], y1 ,n.iter = 100, eta = 0.00015)
result1 <- result1$infomatrix
result2 <- result2$infomatrix
label <- rep("0.00005", dim(result1)[1])
result1 <- cbind(label, result1)
label <- rep("0.00015", dim(result2)[1])
result2 <- cbind(label, result2)
df <- rbind(result1, result2)
ggplot(df, aes(x = No_iteration, y = cost_function)) + 
  geom_line(aes(color=label, linetype=label), size = 1) +
  xlab("No. of iteration") + 
  ylab("Cost Function") +
  ggtitle("Logistic GD - Cost function with various learning rates")


####### Apply to the testing set. ######
y2=testing[,58]
w = result.logGD$w
test.logGD = predict.logistic(w, testing[,-58], y2)
test.logGD




#########################################################################
################################# SVM ###################################
#########################################################################
training.f <- training
testing.f <- testing
training.f$spam <- as.factor(training.f$spam)
testing.f$spam <- as.factor(testing.f$spam)

registerDoMC(cores=5)
### finding optimal value of a tuning parameter
sigDist <- sigest(spam ~ ., data = training.f, frac = 1)
### creating a grid of two tuning parameters, .sigma comes from the earlier line. we are trying to find best value of .C
svmTuneGrid <- data.frame(.sigma = sigDist[1], .C = 2^(-2:7))

## training the SVM model can take some time, running time may exceed 10 minutes 
x <- train(spam ~ ., data = training.f,method = "svmRadial",tuneGrid = svmTuneGrid,trControl = trainControl(method = "repeatedcv", repeats = 5, classProbs =  FALSE))
proc.time()
plot(x, col = "hotpink2", lwd = 1.6, cex = 1)
pred <- predict(x,testing.f[,1:57])
acc <- confusionMatrix(pred,testing.f$spam)

