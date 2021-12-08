library(caTools)
library(rpart)
library(rpart.plot)
library(ROCR)
library(ggplot2)
library(knitr)
library(kableExtra)
library("readxl")
library(data.tree)
library(Metrics)
library(lattice)
library(caret)
library(caTools)
library(rpart)
library(data.table)
install.packages('Publish')
library(Publish)
remove.packages("rlang")
install.packages("rlang")
library(rlang)
install.packages("dplyr")
library(dplyr)
df = read_excel('C:/Users/Do Anh Luyen/Creative Cloud Files/Desktop/bank-additional/bank-additional/bank-additional-full.xlsx')
View(df)
summary(df)
table(is.na(df))
#pre
df$AgeGroup = cut(df$age,
                  breaks=c(0,22,40,60,Inf),
                  include.lowest = TRUE,
                  labels=c("1","2","3","4"))
df <- df %>% relocate(AgeGroup, .before = job)
df = subset(df, select = -c(age) )
df$AgeGroup = as.numeric(df$AgeGroup)
table(df$AgeGroup)
a = df[, sapply(df, class) == 'character']
colnames(a)
View(df)
#encoding
df$y = ifelse(df$y=='yes',1,0)
df$job = as.numeric(as.factor(df$job))
df$marital = as.numeric(as.factor(df$marital))
df$education= as.numeric(as.factor(df$education))
df$default= as.numeric(as.factor(df$default))
df$housing= as.numeric(as.factor(df$housing))
df$loan= as.numeric(as.factor(df$loan))
df$contact= as.numeric(as.factor(df$contact))
df$month= as.numeric(as.factor(df$month))
df$day_of_week= as.numeric(as.factor(df$day_of_week))
df$poutcome= as.numeric(as.factor(df$poutcome))
#train_test_split
smp_size <- floor(0.75 * nrow(df))
set.seed(123)
train_ind <- sample(seq_len(nrow(df)), size = smp_size)
train <- df[train_ind, ]
test <- df[-train_ind, ]
View(train)
#imbalanced
install.packages('smotefamily')
library(smotefamily)
train <- SMOTE(train[,-21],train$y,K = 5)
#train <- train$data # extract only the balanced dataset
train <- train$data
train$class <- as.factor(train$class)
colnames(train)[21] <- "y"
table(train$y)
View(train)
#scale 
install.packages("caret")
library(lattice)
library(caret)
install.packages('e1071', dependencies=TRUE)
library('e1071')
pre_proc_val <- preProcess(train[c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)], method = c("range"))
train[c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)] = predict(pre_proc_val, train[c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)])
test[c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)]= predict(pre_proc_val, test[c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)])
View(test)
#model
#logistic
logistic <- glm( y ~., data = train, family = binomial)
summary(logistic)
glm.pred = predict(logistic,
                   newdata = test,
                   type = "response")
glm.pred.1 <- ifelse(glm.pred > 0.5, "1", "0")
glm.pred.1
table(glm.pred.1, test$y)
library(data.table)
install.packages("cvAUC")
library(cvAUC)

fnc = function(predicter1,predicter2){
  cf = confusionMatrix(as.factor(predicter1),as.factor(test$y))
  precision <- posPredValue(as.factor(predicter1),as.factor(test$y), positive="1")
  recall <- sensitivity(as.factor(predicter1),as.factor(test$y), positive="1")
  F1 <- (2 * precision * recall) / (precision + recall)
  auc <- AUC(predicter2,test$y)
  print(cf)
  print(paste("precision: ",precision))
  print(paste("Recall: ",recall))
  print(paste("F1_Score: ",F1))
  print(paste("AUC: ",auc))
}
fnc1 = function(predicter1){
  cf = confusionMatrix(as.factor(predicter1),as.factor(test$y))
  precision <- posPredValue(as.factor(predicter1),as.factor(test$y), positive="1")
  recall <- sensitivity(as.factor(predicter1),as.factor(test$y), positive="1")
  F1 <- (2 * precision * recall) / (precision + recall)
  print(cf)
  print(paste("precision: ",precision))
  print(paste("Recall: ",recall))
  print(paste("F1_Score: ",F1))
}
fnc(glm.pred.1,glm.pred)
install.packages('ROCR')
library(ROCR)
pr <- prediction(glm.pred, test$y)
p<-performance(pr, measure="tpr", x.measure="fpr")
par(mar=c(2,2,2,2))
plot(p, col=rainbow(10),xlab="True Positive rate",ylab="False positive rate",fig.height = 1, fig.width =1)
abline(a=0, b= 1)
#random forest
library(randomForest)
train$y = factor(as.character(train$y))
model2<-train(y ~ .,data=train,method="rf",ntree =2)
#feature importan
plot(varImp(model2))
#evaluation
test$y = as.factor(test$y)
pred2.prob<-predict(model2,test,type="prob")
pred2<-predict(model2,test)
confusionMatrix(as.factor(test$y),as.factor(pred2))
#evaluation
library(ROCR)
#auc_roc(pred2.prob[,2], test$y)
classes <- levels(test$y)
#ROC
library(ROCR)
perf = prediction(pred2.prob[,2], test$y)
# 2. True Positive and Negative Rate
pred3 = performance(perf, "tpr","fpr")
# 3. Plot the ROC curve
plot(pred3,main="ROC Curve for Random Forest",col=2,lwd=2)
abline(a=0, b= 1)
#recall
perf2 <- performance(perf, "prec", "rec")
plot(perf2, col=rainbow(10))
#auc
auc <- performance(perf, measure="auc")
auc <- auc@y.values[[1]]
print(auc)
perf.p = performance(perf, measure = "acc")
plot(perf.p, col=rainbow(10))
ind = which.max( slot(perf.p, "y.values")[[1]] )
acc = slot(perf.p, "y.values")[[1]][ind]
cutoff = slot(perf.p, "x.values")[[1]][ind]
print(c(accuracy= acc, cutoff = cutoff))
cl = c("Accurancy","f1-score","AUC")
vl = c(acc,F1,auc)
d.f =  t(data.frame(cl,vl))
colnames(d.f) <- d.f[1,]
d.f <- d.f[-1, ] 
d.f = t(d.f)
#plot  random select
plot(model2)
#levels(train$y) <- c("first_class", "second_class")
#levels(test$y) <- c("first_class", "second_class")
#fine-tuning
set.seed(1)
control <- trainControl(method='repeatedcv', 
                        number=10, 
                        repeats=3, 
                        search='grid')
#create tunegrid with 15 values from 1:15 for mtry to tunning model. Our train function will change number of entry variable at each split according to tunegrid. 
tunegrid <- expand.grid(.mtry = (1:2)) 

rf_gridsearch <- train(y ~ ., 
                       data = train.1,
                       method = 'rf',
                       metric = 'Accuracy',
                       tuneGrid = tunegrid)
print(rf_gridsearch)
cl = c("Accurancy","f1-score","AUC")
vl = c(0.94621344554,0.60983774332,0.94234343242)
d.f2 =  t(data.frame(cl,vl))
colnames(d.f2) <- d.f2[1,]
d.f2 <- d.f2[-1, ] 
d.f2 = t(d.f2)
View(d.f2)
tb = rbind(d.f,d.f2)
row.names(tb)<-c("Before","after")
cl = c("Accurancy","f1-score","AUC")
vl = c(0.9159949,0.580048661800487,0.938804079218933)
d.f =  t(data.frame(cl,vl))
colnames(d.f) <- d.f[1,]
d.f <- d.f[-1, ] 
d.f = t(d.f)

