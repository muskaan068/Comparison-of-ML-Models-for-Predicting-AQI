---
  title: "Comparison of ML Models for Predicting AQI in different cities of India"
author: "MUSKAAN YADAV"
date: "2023-03-19"
output: html_document
---
  
  ```{r}
library(dplyr)
#for logistic regression
library(caTools)
#for decision tree and naive bayes
library(e1071)
library(party)
# Load rpart
library(rpart)
library(rpart.plot)
#for random forest
library(randomForest)
library(ggplot2)

```

```{r}
df <- read.csv('state_weather_aqi_data_mf2.csv')

#dataset description
str(df)
dim(df)
colnames(df)
```


```{r}
split=sample.split(df,SplitRatio=0.8)
train_reg=subset(df,split=="TRUE")
test_reg=subset(df,split=="FALSE")
```





```{r}
#Multiple Linear Regression

mLinearReg <- lm(AQI~PM2.5+PM10+NO2+NH3+SO2+CO+OZONE, data=df)
#summary(mLinearReg)
#plot(mLinearReg)
predict_ml=predict(mLinearReg,test_reg)




```


```{r}
new=test_reg$AQI
newtest=as.numeric(new)

```


```{r}
library(Metrics)
maemlr<-mae(predict_ml,newtest)
rmsemlr<-rmse(predict_ml,newtest)
rmslemlr<-rmsle(predict_ml,newtest)
rsquaremlr<-(cor(predict_ml,newtest))**2
rsquaremlr
```


```{r}
#Decision Tree
dmodel=ctree(AQI~PM2.5+PM10+NO2+NH3+SO2+CO+OZONE,data=train_reg)
plot(dmodel, main = "My Decision Tree", fig.width = 100, fig.height = 50)

predict_AQI<-predict(dmodel,test_reg)
predict_AQI


```


```{r}
#Decision tree Accuracy
maedt<-mae(predict_AQI,newtest)
rmsedt<-rmse(predict_AQI,newtest)
rmsledt<-rmsle(predict_AQI,newtest)
rsquare<-(cor(predict_AQI,newtest))**2
rsqaured<-rsquare[1]
rsqaured

```


```{r}
#Random Forest
set.seed(120)
#classifier_RF = randomForest(x = train_reg[-5],
#y = train_reg$AQI,
#ntree = 500)
classifier_RF=randomForest(AQI~PM2.5+PM10+NO2+NH3+SO2+CO+OZONE,data=train_reg,ntree=500)

classifier_RF
```

```{r}
#y_pred = predict(classifier_RF, newdata = test_reg[-5])
df1=subset(test_reg,select=c(PM2.5,PM10,NO2,NH3,SO2,CO,OZONE,AQI))
y_pred=predict(classifier_RF,df1)

maerf<-mae(y_pred,newtest)
rmserf<-rmse(y_pred,newtest)
rmslerf<-rmsle(y_pred,newtest)
rsquare<-(cor(y_pred,newtest))**2
rsquarerf<-rsquare[1]
rsquarerf

```

```{r}
#Lasso regression
library(glmnet)
y<-train_reg$AQI
x<-data.matrix(train_reg[,c('PM2.5','PM10','NO2','NH3','SO2','CO','OZONE')])
lasso_model<-cv.glmnet(x,y,alpha=1)  #alpha=1,lasso alpha=0 ridge, alpha=0-1, elastic net
best_lambda<-lasso_model$lambda.min
best_lambda
plot(lasso_model)


```

```{r}
best_model<-glmnet(x,y,alpha=1,lambda=best_lambda)
#coef(best_model)
lass_test=data.matrix(test_reg[,c('PM2.5','PM10','NO2','NH3','SO2','CO','OZONE')])
lasso_pred<-predict(best_model,s=best_lambda,newx=lass_test)
mael<-mae(lasso_pred,newtest)
rmsel<-rmse(lasso_pred,newtest)
rmslel<-rmsle(lasso_pred,newtest)
rsquare<-(cor(lasso_pred,newtest))**2
rsquarel<-rsquare[1]
rsquarel



```


```{r}
#Polynomial Regression
K<-10  #number of folds to use for k-fold cross valid.
degree<-5  #max degree to find best optimal degree of poly.
df.shuffled<-df[sample(nrow(df)),]
folds<-cut(seq(1,nrow(df.shuffled)),breaks=K,labels=FALSE)
mse=matrix(data=NA,nrow=K,ncol=degree)

#considering PM2.5 as dominant factor
for(i in 1:K){
  #training and test data
  testIndexes<-which(folds==i,arr.ind=TRUE)
  testData<-df.shuffled[testIndexes,]
  trainData<-df.shuffled[-testIndexes,]
  
  #use k-fold cv to evaluate models
  for(j in 1:degree){
    fit.train=lm(AQI~poly(PM2.5,j),data=trainData)
    fit.test=predict(fit.train,newdata=testData)
    mse[i,j]=mean((fit.test-testData$AQI)^2)
  }
}

pm2.5mse<-colMeans(mse)
#considering PM10 as dominant factor
for(i in 1:K){
  #training and test data
  testIndexes<-which(folds==i,arr.ind=TRUE)
  testData<-df.shuffled[testIndexes,]
  trainData<-df.shuffled[-testIndexes,]
  
  #use k-fold cv to evaluate models
  for(j in 1:degree){
    fit.train=lm(AQI~poly(PM10,j),data=trainData)
    fit.test=predict(fit.train,newdata=testData)
    mse[i,j]=mean((fit.test-testData$AQI)^2)
  }
}

pm10mse<-colMeans(mse)

#considering PM10 as dominant factor
for(i in 1:K){
  #training and test data
  testIndexes<-which(folds==i,arr.ind=TRUE)
  testData<-df.shuffled[testIndexes,]
  trainData<-df.shuffled[-testIndexes,]
  
  #use k-fold cv to evaluate models
  for(j in 1:degree){
    fit.train=lm(AQI~poly(NO2,j),data=trainData)
    fit.test=predict(fit.train,newdata=testData)
    mse[i,j]=mean((fit.test-testData$AQI)^2)
  }
}

pmno2mse<-colMeans(mse)

for(i in 1:K){
  #training and test data
  testIndexes<-which(folds==i,arr.ind=TRUE)
  testData<-df.shuffled[testIndexes,]
  trainData<-df.shuffled[-testIndexes,]
  
  #use k-fold cv to evaluate models
  for(j in 1:degree){
    fit.train=lm(AQI~poly(NH3,j),data=trainData)
    fit.test=predict(fit.train,newdata=testData)
    mse[i,j]=mean((fit.test-testData$AQI)^2)
  }
}

pmnh3mse<-colMeans(mse)

for(i in 1:K){
  #training and test data
  testIndexes<-which(folds==i,arr.ind=TRUE)
  testData<-df.shuffled[testIndexes,]
  trainData<-df.shuffled[-testIndexes,]
  
  #use k-fold cv to evaluate models
  for(j in 1:degree){
    fit.train=lm(AQI~poly(SO2,j),data=trainData)
    fit.test=predict(fit.train,newdata=testData)
    mse[i,j]=mean((fit.test-testData$AQI)^2)
  }
}

so2mse<-colMeans(mse)

for(i in 1:K){
  #training and test data
  testIndexes<-which(folds==i,arr.ind=TRUE)
  testData<-df.shuffled[testIndexes,]
  trainData<-df.shuffled[-testIndexes,]
  
  #use k-fold cv to evaluate models
  for(j in 1:degree){
    fit.train=lm(AQI~poly(CO,j),data=trainData)
    fit.test=predict(fit.train,newdata=testData)
    mse[i,j]=mean((fit.test-testData$AQI)^2)
  }
}

comse<-colMeans(mse)

for(i in 1:K){
  #training and test data
  testIndexes<-which(folds==i,arr.ind=TRUE)
  testData<-df.shuffled[testIndexes,]
  trainData<-df.shuffled[-testIndexes,]
  
  #use k-fold cv to evaluate models
  for(j in 1:degree){
    fit.train=lm(AQI~poly(OZONE,j),data=trainData)
    fit.test=predict(fit.train,newdata=testData)
    mse[i,j]=mean((fit.test-testData$AQI)^2)
  }
}

ozonemse<-colMeans(mse)


pm2.5mse
pm10mse
pmno2mse
pmnh3mse
so2mse
comse
ozonemse
```
```{r}
#fit best model i.e. is PM2.5 as it has least errors and that too with degree 4
best=lm(AQI~poly(PM2.5,3,raw=TRUE),data=df)
df2=subset(test_reg,select=c(PM2.5))
predicted_val_poly<-predict(best,df2)
summ=summary(best)
maep<-mae(predicted_val_poly,newtest)
rmsep<-rmse(newtest,predicted_val_poly)
rmslep<-rmsle(newtest,predicted_val_poly)
rsquarep<-summ$r.squared[1]
```

```{r}
ggplot(df,aes(x=PM2.5,y=AQI))+
  geom_point()+
  stat_smooth(method='lm',formula=y~poly(x,3),size=1)+
  xlab("PM2.5")+
  ylab("AQI")
```



```{r}
ggplot(df,aes(x=PM10,y=AQI))+
  geom_point()+
  stat_smooth(method='lm',formula=y~poly(x,3),size=1)+
  xlab("PM10")+
  ylab("AQI")
```


```{r}
ggplot(df,aes(x=NO2,y=AQI))+
  geom_point()+
  stat_smooth(method='lm',formula=y~poly(x,3),size=1)+
  xlab("NO2")+
  ylab("AQI")
```


```{r}
ggplot(df,aes(x=NH3,y=AQI))+
  geom_point()+
  stat_smooth(method='lm',formula=y~poly(x,3),size=1)+
  xlab("NH3")+
  ylab("AQI")
```


```{r}
ggplot(df,aes(x=SO2,y=AQI))+
  geom_point()+
  stat_smooth(method='lm',formula=y~poly(x,3),size=1)+
  xlab("SO2")+
  ylab("AQI")
```

```{r}
ggplot(df,aes(x=CO,y=AQI))+
  geom_point()+
  stat_smooth(method='lm',formula=y~poly(x,3),size=1)+
  xlab("CO")+
  ylab("AQI")
```

```{r}
ggplot(df,aes(x=OZONE,y=AQI))+
  geom_point()+
  stat_smooth(method='lm',formula=y~poly(x,3),size=1)+
  xlab("OZONE")+
  ylab("AQI")
```


```{r}
#Tabulating the matrix
findf<-data.frame(
  models=c('MLR','Decision Tree','Random Forest','Lasso Regression','Polynomial Regression'),
  Rsqaure=c(rsquaremlr,rsqaured,rsquarerf,rsquarel,rsquarep),
  MAE=c(maemlr,maedt,maerf,mael,maep),
  RMSE=c(rmsemlr,rmsedt,rmserf,rmsel,rmsep),
  RMSLE=c(rmslemlr,rmsledt,rmslemlr,rmslel,rmslep)
  
  
)
findf
```

