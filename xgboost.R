#load libraries
library(data.table)
library(mlr)

setcol <- c("age","workclass","fnlwgt","education","education-num","marital-status",
            "occupation","relationship","race","sex","capital-gain","capital-loss",
            "hours-per-week","native-country","target")

#load data
train <- read.table("C:/Users/sanjay/Desktop/kaggle/machine learning/adult.data.txt",
                    header = F, sep = ",", col.names = setcol, na.strings = c(" ?"), 
                    stringsAsFactors = F)
test <- read.table("C:/Users/sanjay/Desktop/kaggle/machine learning/adult.test.txt",
                   header = F,sep = ",",col.names = setcol,skip = 1, na.strings = c(" ?"),
                   stringsAsFactors = F)

#convert data frame to data table
setDT(train) 
setDT(test)

#check missing values 
table(is.na(train))
sapply(train, function(x) sum(is.na(x))/length(x))*100


#quick data cleaning
#remove extra character from target variable
library(stringr)
test [,target := substr(target,start = 1,stop = nchar(target)-1)]

#remove leading whitespaces
char_col <- colnames(train)[ sapply (test,is.character)]
for(i in char_col) set(train,j=i,value = str_trim(train[[i]],side = "left"))
for(i in char_col) set(test,j=i,value = str_trim(test[[i]],side = "left"))


#set all missing value as "Missing" 
train[is.na(train)] <- "Missing" 
test[is.na(test)] <- "Missing"

#using one hot encoding 
labels <- train$target 
ts_label <- test$target
new_tr <- model.matrix(~.+0,data = train[,-c("target"),with=F]) 
new_ts <- model.matrix(~.+0,data = test[,-c("target"),with=F])

#convert factor to numeric 
labels <- as.factor(labels)
ts_label <- as.factor(ts_label)
labels <- as.numeric(labels)-1
ts_label <- as.numeric(ts_label)-1

library(xgboost)
#preparing matrix 
dtrain <- xgb.DMatrix(data = new_tr,label = labels) 
dtest <- xgb.DMatrix(data = new_ts,label=ts_label)

#default parameters
params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.3, 
               gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)

xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 100, nfold = 5, 
                 showsd = T, stratified = T, print_every_n = 10, early_stopping_rounds = 20, 
                 maximize = F)

#first default - model training
xgb1 <- xgb.train (params = params, data = dtrain, nrounds = 62, watchlist = list(val=dtest,train=dtrain),
                   print_every_n = 10, early_stopping_round = 10, maximize = F , eval_metric = "error")

#model prediction
xgbpred <- predict (xgb1,dtest)
xgbpred <- ifelse (xgbpred > 0.5,1,0)

#confusion matrix
library(caret)
confusionMatrix (xgbpred, ts_label)
#Accuracy - 87.34%` 

        
        ###########################Xgboost R-tutorial ################################
        ##############################################################################

drat:::addRepo("dmlc")
require(xgboost)

#load test and train data in r
data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
train <- agaricus.train
test <- agaricus.test


# Each variable is a list containing two things, label and data:
str(train)

# check the dimension of test and train data
dim(train$data)
dim(test$data)

# As seen below, the data are stored in a dgCMatrix which is a sparse matrix and label vector
# is a numeric vector ({0,1}):
  
class(train$data)[1]
## [1] "dgCMatrix"
class(train$label)
## [1] "numeric"


#   We will train decision tree model using the following parameters:
  
# objective = "binary:logistic": we will train a binary classification model ;
# max_depth = 2: the trees won't be deep, because our case is very simple ;
# nthread = 2: the number of cpu threads we are going to use;
# nrounds = 2: there will be two passes on the data, the second one will enhance
# the model by further reducing the difference between ground truth and prediction.


bstSparse <- xgboost(data = train$data, label = train$label, max_depth = 2, eta = 1, 
                     nthread = 2, nrounds = 2, objective = "binary:logistic")
##  Dense matrix
##  Alternatively, you can put your dataset in a dense matrix, i.e. a basic R matrix.
bstDense <- xgboost(data = as.matrix(train$data), label = train$label, max_depth = 2,
                    eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")

## xgb.DMatrix
## XGBoost offers a way to group them in a xgb.DMatrix. You can even add other meta data in 
## it. It will be useful for the most advanced features we will discover later.
dtrain <- xgb.DMatrix(data = train$data, label = train$label)
bstDMatrix <- xgboost(data = dtrain, max_depth = 2, eta = 1, nthread = 2, nrounds = 2, 
                      objective = "binary:logistic")

## Verbose option
## XGBoost has several features to help you to view how the learning progress internally. 
## The purpose is to help you to set the best parameters, which is the key of your model quality.

# verbose = 0, no message
bst <- xgboost(data = dtrain, max_depth = 2, eta = 1, nthread = 2, nrounds = 2, 
               objective = "binary:logistic", verbose = 0)

#verbose = 2, also print information about tree
bst <- xgboost(data = dtrain, max_depth = 2, eta = 1, nthread = 2, nrounds = 2, 
               objective = "binary:logistic", verbose = 2)


##  Basic prediction using XGBoost. Perform the prediction
pred <- predict(bst, test$data)

# size of the prediction vector
length(pred)

# limit display of predictions to the first 10
head(pred)

#[1] 0.2858301699 0.9239239097 0.2858301699 0.2858301699 0.0516987294 0.9239239097

#  These numbers doesn't look like binary classification {0,1}. We need to perform a 
#  simple transformation before being able to use these results.

## Transform the regression in a binary classification
## The only thing that XGBoost does is a regression. XGBoost is using label vector to build
## its regression model.

prediction <- as.numeric(pred > 0.5)
head(prediction)

##[1] 0 1 0 0 0 1

## Measuring model performance

err <- mean(as.numeric(pred > 0.5) != test$label)
print(paste("test-error=", err))
# [1] "test-error= 0.0217256362507759"

## Note that the algorithm has not seen the test data during the model construction.

## Steps explanation:
## 1) as.numeric(pred > 0.5) applies our rule that when the probability (<=> regression <=> prediction)
## is > 0.5 the observation is classified as 1 and 0 otherwise ;
## 2) probabilityVectorPreviouslyComputed != test$label computes the vector of error between true data
## and computed probabilities ;
## 3) mean(vectorOfErrors) computes the average error itself.

## The most important thing to remember is that to do a classification, you just do a regression to 
## the label and then apply a threshold.

## Dataset Preparation
dtrain <- xgb.DMatrix(data = train$data, label=train$label)
dtest <- xgb.DMatrix(data = test$data, label=test$label)

## Measure learning progress with xgb.train

## Both xgboost (simple) and xgb.train (advanced) functions train models.

## One of the special feature of xgb.train is the capacity to follow the progress of the learning 
## after each round. Because of the way boosting works, there is a time when having too many rounds 
## lead to an overfitting. You can see this feature as a cousin of cross-validation method. The following 
## techniques will help you to avoid overfitting or optimizing the learning time in stopping it as soon
## as possible.

## One way to measure progress in learning of a model is to provide to XGBoost a second dataset already
## classified. Therefore it can learn on the first dataset and test its model on the second one. 
## Some metrics are measured after each round during the learning.

watchlist <- list(train=dtrain, test=dtest)

bst <- xgb.train(data=dtrain, max_depth=2, eta=1, nthread = 2, nrounds=2, watchlist=watchlist,
                 objective = "binary:logistic")

## XGBoost has computed at each round the same average error metric than seen above 
## (we set nrounds to 2, that is why we have two lines). Obviously, the train-error number is 
## related to the training dataset (the one the algorithm learns from) and the test-error number
## to the test dataset.

## Both training and test error related metrics are very similar, and in some way, it makes sense: 
## what we have learned from the training dataset matches the observations from the test dataset.

## If with your own dataset you have not such results, you should think about how you divided your 
## dataset in training and test. May be there is something to fix. Again, caret package may help.

## For a better understanding of the learning progression, you may want to have some specific metric
## or even use multiple evaluation metrics.

bst <- xgb.train(data=dtrain, max_depth=2, eta=1, nthread = 2, nrounds=2, 
                 watchlist=watchlist, eval_metric = "error", eval_metric = "logloss", 
                 objective = "binary:logistic")

## [1]	train-error:0.046522	train-logloss:0.233376	test-error:0.042831	test-logloss:0.226686 
## [2]	train-error:0.022263	train-logloss:0.136658	test-error:0.021726	test-logloss:0.137874 
# eval_metric allows us to monitor two new metrics for each round, logloss and error. 


                                     ## Linear boosting
## Until now, all the learnings we have performed were based on boosting trees.
## XGBoost implements a second algorithm, based on linear boosting. The only difference with 
## previous command is booster = "gblinear" parameter (and removing eta parameter).

bst <- xgb.train(data=dtrain, booster = "gblinear", max_depth=2, nthread = 2, nrounds=2,
                 watchlist=watchlist, eval_metric = "error", eval_metric = "logloss",
                 objective = "binary:logistic")

## [1]	train-error:0.023184	train-logloss:0.181862	test-error:0.017381	test-logloss:0.180453 
## [2]	train-error:0.004453	train-logloss:0.069127	test-error:0.003724	test-logloss:0.067083 
# In this specific case, linear boosting gets sligtly better performance metrics than decision
#  trees based algorithm.

## Manipulating xgb.DMatrix

## Save / Load
xgb.DMatrix.save(dtrain, "dtrain.buffer")


# to load it in, simply call xgb.DMatrix
dtrain2 <- xgb.DMatrix("dtrain.buffer")

bst <- xgb.train(data=dtrain2, max_depth=2, eta=1, nthread = 2, nrounds=2, 
                 watchlist=watchlist, objective = "binary:logistic")

## Information extraction

label = getinfo(dtest, "label")
pred <- predict(bst, dtest)
err <- as.numeric(sum(as.integer(pred > 0.5) != label))/length(label)
print(paste("test-error=", err))

## View feature importance/influence from the learnt model

importance_matrix <- xgb.importance(model = bst)
print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix)

## View the trees from a model

xgb.dump(bst, with_stats = T)

library(DiagrammeR)
xgb.plot.tree(model = bst)

# if you provide a path to fname parameter you can save the trees to your hard drive.

## Save and load models

# save model to binary local file
xgb.save(bst, "xgboost.model")

#xgb.save function should return TRUE if everything goes well and crashes otherwise.

# load binary model to R
bst2 <- xgb.load("xgboost.model")
pred2 <- predict(bst2, test$data)

# And now the test
print(paste("sum(abs(pred2-pred))=", sum(abs(pred2-pred))))

# save model to R's raw vector
rawVec <- xgb.save.raw(bst)

# print class
print(class(rawVec))

# load binary model to R
bst3 <- xgb.load(rawVec)
pred3 <- predict(bst3, test$data)

# pred2 should be identical to pred
print(paste("sum(abs(pred3-pred))=", sum(abs(pred2-pred))))


















