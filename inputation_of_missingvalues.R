
# read the data into R
dat <- read.csv(url("https://goo.gl/4DYzru"), header=TRUE, sep=",")

# looking at the few observations in R
head(dat)

# Check the data for missing values
sapply(dat, function(x) sum(is.na(x)))

original <- dat

# Now I will add some missings in few variables.

set.seed(10)
dat[sample(1:nrow(dat), 20), "Cholesterol"] <- NA
dat[sample(1:nrow(dat), 20), "Smoking"] <- NA
dat[sample(1:nrow(dat), 20), "Education"] <- NA
dat[sample(1:nrow(dat), 5), "Age"] <- NA
dat[sample(1:nrow(dat), 5), "BMI"] <- NA

# Check the data for missing values
sapply(dat, function(x) sum(is.na(x)))

#mice package for imputation of missing values, do not change anything but the data
library(mice)
init = mice(dat, maxit=0) 
meth = init$method
predM = init$predictorMatrix

# The code below will remove the variable as a predictor but still will be imputed. 
# Just for illustration purposes, I select the BMI variable to not be included as predictor
# during imputation.
predM[, c("BMI")]=0


# If you want to skip a variable from imputation use the code below. 
# This variable will be used for prediction.
meth[c("Age")]=""


# Now let specify the methods for imputing the missing values. 
# There are specific methods for continuous, binary and ordinal variables.
# I set different methods for each variable. 
# You can add more than one variable in each method.

meth[c("Cholesterol")]="norm" 
meth[c("Smoking")]="logreg" 
meth[c("Education")]="polyreg"

# Now it is time to run the multiple (m=5) imputation.
set.seed(103)
imputed = mice(dat, method=meth, predictorMatrix=predM, m=5)

imputed <- complete(imputed)

# Check the data for missing values
sapply(imputed, function(x) sum(is.na(x)))

#check the accuracy of the missing varaibles
# Cholesterol
actual <- original$Cholesterol[is.na(dat$Cholesterol)]
predicted <- imputed$Cholesterol[is.na(dat$Cholesterol)]
table(actual)
table(predicted)
mean(actual)
mean(predicted)
# Smoking
actual <- original$Smoking[is.na(dat$Smoking)] 
predicted <- imputed$Smoking[is.na(dat$Smoking)] 
table(actual)
table(predicted)

########################################################################
########################################################################
########################################################################

# initialize the data
data ("BostonHousing", package="mlbench")
original <- BostonHousing  # backup original data

# Introduce missing values
set.seed(100)
BostonHousing[sample(1:nrow(BostonHousing), 40), "rad"] <- NA
BostonHousing[sample(1:nrow(BostonHousing), 40), "ptratio"]<- NA     

# Pattern of missing values
library(mice)
md.pattern(BostonHousing)  # pattern or missing values in data.
##      crim zn indus chas nox rm age dis tax b lstat medv rad  ptratio   
## 431    1  1     1    1   1  1   1   1   1  1     1    1   1       1  0
## 35     1  1     1    1   1  1   1   1   1  1     1    1   0       1  1
## 35     1  1     1    1   1  1   1   1   1  1     1    1   1       0  1
## 5      1  1     1    1   1  1   1   1   1  1     1    1   0       0  2
##        0  0     0    0   0  0   0   0   0  0     0    0  40      40 80

# From this we can see that there are exactly 431 observations which aren't missing 
# any values. There are 35 observations that are missing rad, 35 observations that are
# missing ptratio, and 5 observations that are missing both the rad and the ptratio.So
# there are a total of 80 observations that are missing atleast 1 value in the data.

# 1. deleting the observations
lm(medv ~ ptratio + rad, data=BostonHousing, na.action=na.omit)

# 2. we can also delete the variable if lots of them are missing

# 3. imputation with mean, median or mode
library(Hmisc)
impute(BostonHousing$ptratio, mean)  # replace with mean
impute(BostonHousing$ptratio, median)  # median
impute(BostonHousing$ptratio, 20)  # replace specific number

# compute the accuracy when it is imputed with mean
library(DMwR)
actuals <- original$ptratio[is.na(BostonHousing$ptratio)]
predicteds <- rep(mean(BostonHousing$ptratio, na.rm=T), length(actuals))
regr.eval(actuals, predicteds)

# Prediction is most advanced method to impute your missing values and 
# includes different approaches such as: kNN Imputation, rpart, and mice.

########################
# 4.1. kNN Imputation
########################

# For every observation to be imputed, it identifies 'k' closest observations 
# based on the euclidean distance and computes the weighted average (weighted based 
# on distance) of these 'k' obs.


library(DMwR)
knnOutput <- knnImputation(BostonHousing[, !names(BostonHousing) %in% "medv"]) 
# perform knn imputation.
anyNA(knnOutput)

# compute the accuracy
actuals <- original$ptratio[is.na(BostonHousing$ptratio)]
predicteds <- knnOutput[is.na(BostonHousing$ptratio), "ptratio"]
regr.eval(actuals, predicteds)
##  mae        mse       rmse       mape 
##  1.00188715 1.97910183 1.40680554 0.05859526 

#################
# 4.1.  rpart
#################

# The limitation with DMwR::knnImputation is that it sometimes may 
# not be appropriate to use when the missing value comes from a factor 
# variable. Both rpart and mice has flexibility to handle that scenario. 
# The advantage with rpart is that you just need only one of the variables
# to be non NA in the predictor fields.. To handle factor variable, we can 
# set the method=class while calling rpart(). For numeric, we use, method=anova.
# Here again, we need to make sure not to train rpart on response variable (medv).

library(rpart)
class_mod <- rpart(rad ~ . - medv, data=BostonHousing[!is.na(BostonHousing$rad), ], 
                   method="class", na.action=na.omit)  # since rad is a factor
anova_mod <- rpart(ptratio ~ . - medv, data=BostonHousing[!is.na(BostonHousing$ptratio), ],
                   method="anova", na.action=na.omit)  # since ptratio is numeric.
rad_pred <- predict(class_mod, BostonHousing[is.na(BostonHousing$rad), ])
ptratio_pred <- predict(anova_mod, BostonHousing[is.na(BostonHousing$ptratio), ])

# compute the accuracy
actuals <- original$ptratio[is.na(BostonHousing$ptratio)]
predicteds <- ptratio_pred
regr.eval(actuals, predicteds)
##  mae        mse       rmse       mape 
##  0.71061673 0.99693845 0.99846805 0.04099908 

# Accuracy for rad
actuals <- original$rad[is.na(BostonHousing$rad)]
predicteds <- as.numeric(colnames(rad_pred)[apply(rad_pred, 1, which.max)])
mean(actuals != predicteds)  # compute misclass error.
## [1] 0.25
# This yields a mis-classification error of 25%. Not bad for a factor variable!

#######################
# 4.3 mice
#######################

# mice short for Multivariate Imputation by Chained Equations is an R package that 
# provides advanced features for missing value treatment.

library(mice)
miceMod <- mice(BostonHousing[, !names(BostonHousing) %in% "medv"], method="rf")  
# perform mice imputation, based on random forests.
miceOutput <- complete(miceMod)  # generate the completed data.
anyNA(miceOutput)

# compute the accuracy
actuals <- original$ptratio[is.na(BostonHousing$ptratio)]
predicteds <- miceOutput[is.na(BostonHousing$ptratio), "ptratio"]
regr.eval(actuals, predicteds)
##  mae        mse       rmse       mape 
##  0.88250000 3.31225000 1.81995879 0.05378101 

# Accuracy for rad
actuals <- original$rad[is.na(BostonHousing$rad)]
predicteds <- miceOutput[is.na(BostonHousing$rad), "rad"]
mean(actuals != predicteds)  # compute misclass error.
## [1] 0.225

                       ########################################
          ### Imputing Missing Data with R; MICE package- Visualization ###
                       ########################################


data <- airquality
data[4:10,3] <- rep(NA,7)
data[1:5,4] <- NA

# As far as categorical variables are concerned, replacing categorical variables is
# usually not advisable. Some common practice include replacing missing categorical 
# variables with the mode of the observed ones, however, it is questionable whether 
# it is a good choice. Even though in this case no datapoints are missing from the 
# categorical variables, we remove them from our dataset 
data <- data[-c(5,6)]
summary(data)

##    There are two types of missing data:
# MCAR: missing completely at random. This is the desirable scenario in case of
#       missing data.
# MNAR: missing not at random. Missing not at random data is a more serious issue
#       and in this case it might be wise to check the data gathering process further and 
#       try to understand why the information is missing. For instance, if most of the people
#       in a survey did not answer a certain question, why did they do that? Was the question
#       unclear?

# Assuming data is MCAR, too much missing data can be a problem too.
# Usually a safe maximum threshold is 5% of the total for large datasets. 
# If missing data for a certain feature or sample is more than 5% then you 
# probably should leave that feature or sample out. 
pMiss <- function(x){sum(is.na(x))/length(x)*100}
apply(data,2,pMiss)
apply(data,1,pMiss)

##    Ozone   Solar.R      Wind      Temp 
## 24.183007  4.575163  4.575163  3.267974 

# We see that Ozone is missing almost 25% of the datapoints, therefore 
# we might consider either dropping it from the analysis or gather more measurements. 

# Using mice for looking at missing data pattern

library(mice)
md.pattern(data)

##      Temp Solar.R Wind Ozone   
## 104    1       1    1     1  0
## 34     1       1    1     0  1
## 4      1       0    1     1  1
## 3      1       1    0     1  1
## 3      0       1    1     1  1
## 1      1       0    1     0  2
## 1      1       1    0     0  2
## 1      1       0    0     1  2
## 1      0       1    0     1  2
## 1      0       0    0     0  4
##        5       7    7    37 56

library(VIM)
aggr_plot <- aggr(data, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, 
                  labels=names(data), cex.axis=.7, gap=3,
                  ylab=c("Histogram of missing data","Pattern"))

# 67.97% of the data are not missing any variables, while o.65% of the data 
# are missing all the four variables, ozone, solarR,wind and temp.

marginplot(data[c(1,2)])

#  The red box plot on the left shows the distribution of Solar.
#  R with Ozone missing while the blue box plot shows the distribution
#  of the remaining datapoints. Likewhise for the Ozone box plots at the 
#  bottom of the graph.If our assumption of MCAR data is correct,
#  then we expect the red and blue box plots to be very similar.

# Imputing the missing data

tempData <- mice(data,m=5,maxit=50,meth='pmm',seed=500)
summary(tempData)

# A couple of notes on the parameters:
# m=5 refers to the number of imputed datasets.Five is the default value.
# meth='pmm' refers to the imputation method. In this case we 
# are using predictive mean matching as imputation method. Other 
# imputation methods can be used, type methods(mice) for a list
# of the available imputation methods.

methods(mice)

# If you would like to check the imputed data, for instance for the 
# variable Ozone, you need to enter the following line of code

tempData$imp$Ozone

#       1  2   3   4   5
# 5    13 20  28  12   9
# 10    7 16  28  14  20
# 25    8 14  14   1   8
# 26    9 19  32   8  37
# 27    4 13  10  18  14
# 32   16 65  46  46  40

# The output shows the imputed data for each observation (first column left) 
# within each imputed dataset (first row at the top).

# if we need to check the method of imputation
tempData$meth
## Ozone Solar.R    Wind    Temp 
## "pmm"   "pmm"   "pmm"   "pmm" 

# we can get the completed dataset using the complete function, where 1
# represents the first of the five imputed datasets.
completedData <- complete(tempData,1)

# Inspecting the distribution of original and imputed data

xyplot(tempData,Ozone ~ Wind+Temp+Solar.R,pch=18,cex=1)
# What we would like to see is that the shape of the magenta points 
# (imputed) matches the shape of the blue ones (observed). 
# The matching shape tells us that the imputed values are indeed 
# "plausible values".

densityplot(tempData)
# The density of the imputed data for each imputed dataset is showed in 
# magenta while the density of the observed data is showed in blue. Again, 
# under our previous assumptions we expect the distributions to be similar.

stripplot(tempData, pch = 20, cex = 1.2)
# stripplot() function that shows the distributions of the variables as 
# individual points

                      #####################################                
                      ###########  Pooling  ###############

# Suppose that the next step in our analysis is to fit a linear model to the
# data. You may ask what imputed dataset to choose. The mice package makes it 
# again very easy to fit a a model to each of the imputed dataset and then 
# pool the results together

modelFit1 <- with(tempData,lm(Temp~ Ozone+Solar.R+Wind))
summary(pool(modelFit1))

# The variable modelFit1 containts the results of the fitting performed 
# over the imputed datasets, while the pool() function pools them all together.
# Apparently, only the Ozone variable is statistically significant.

tempData2 <- mice(data,m=50,seed=245435)
modelFit2 <- with(tempData2,lm(Temp~ Ozone+Solar.R+Wind))
summary(pool(modelFit2))

# After having taken into account the random seed initialization, we obtain
# (in this case) more or less the same results as before with only Ozone showing
# statistical significance.


 ######################################################################
 ########## List of R packages for imputing missing values ############
 ######################################################################

## MICE
## Amelia
## missForest
## Hmisc
## mi

# Hmisc should be your first choice of missing value imputation followed by 
# missForest and MICE.Hmisc automatically recognizes the variables types and 
# uses bootstrap sample and predictive mean matching to impute missing values.
# You don't need to separate or treat categorical variable, just like we did
# while using MICE package. However, missForest can outperform Hmisc if the 
# observed variables supplied contain sufficient information.


# I will be using the iris data

library(missForest)

#Get summary
summary(iris)


#Generate 10% missing values at Random 
iris.mis <- prodNA(iris, noNA = 0.1)

#Check missing values introduced in the data
summary(iris.mis)

# I've removed categorical variable. Let's here focus on continuous values.
#remove categorical variables
iris.mis <- subset(iris.mis, select = -c(Species))
summary(iris.mis)


          ###################### MICE ###########################
          #######################################################

#install MICE
library(mice)

md.pattern(iris.mis)

##        Sepal.Length Petal.Width Sepal.Width Petal.Length   
# 97            1           1           1            1  0
# 9             0           1           1            1  1
# 9             1           1           0            1  1
# 17            1           1           1            0  1
# 9             1           0           1            1  1
# 1             0           1           0            1  2
# 4             1           1           0            0  2
# 1             0           0           1            1  2
# 2             1           0           0            1  2
# 1             1           0           1            0  2
#               11          13          16           22 62

library(VIM)
mice_plot <- aggr(iris.mis, col=c('navyblue','yellow'),
                    numbers=TRUE, sortVars=TRUE,
                    labels=names(iris.mis), cex.axis=.7,
                    gap=3, ylab=c("Missing data","Pattern"))

# lets impute the missing values

imputed_Data <- mice(iris.mis, m=5, maxit = 50, method = 'pmm', seed = 500)
summary(imputed_Data)

## Here is an explanation of the parameters used:
# m  - Refers to 5 imputed data sets
# maxit - Refers to no. of iterations taken to impute missing values
# method - Refers to method used in imputation. we used predictive mean matching.

#check imputed values
imputed_Data$imp$Sepal.Width

#get complete data ( 2nd out of 5)
completeData <- complete(imputed_Data,2)

# Also, if you wish to build models on all 5 datasets, you can do it in one 
# go using with() command. You can also combine the result from these models 
# and obtain a consolidated output using pool() command.

#build predictive model
fit <- with(imputed_Data,lm(Sepal.Width ~ Sepal.Length +
                                          Petal.Width)) 

#combine results of all 5 models
summary(pool(fit))



       ###################### AMELIA #########################
       #######################################################

# This package also performs multiple imputation (generate imputed data sets)
# to deal with missing values. Multiple imputation helps to reduce bias and 
# increase efficiency.  It is enabled with bootstrap based EMB algorithm which
# makes it faster and robust to impute many variables including cross sectional,
# time series data etc. Also, it is enabled with parallel imputation feature 
# using multicore CPUs.

##   It makes the following assumptions:
#  All variables in a data set have Multivariate Normal Distribution (MVN).It 
#uses means and covariances to summarize data.
#  Missing data is random in nature (Missing at Random)

library(Amelia)

# The only thing that you need to be careful about is classifying variables. 
# It has 3 parameters:
  
#  idvars - keep all ID variables and other variables which you don't want 
#to impute
#  noms - keep nominal variables here

#seed 10% missing values
iris.mis <- prodNA(iris, noNA = 0.1)
summary(iris.mis)

#specify columns and run amelia
amelia_fit <- amelia(iris.mis, m=5, parallel = "multicore", noms = "Species")

#access imputed outputs
amelia_fit$imputations[[1]]
amelia_fit$imputations[[2]]
amelia_fit$imputations[[3]]
amelia_fit$imputations[[4]]
amelia_fit$imputations[[5]]

# To check a particular column in a data set, use the following commands
amelia_fit$imputations[[5]]$Sepal.Length

#export the outputs to csv files
write.amelia(amelia_fit, file.stem = "imputed_data_set")

        ###################### missForest #########################
        ###########################################################

#   It's a non parametric imputation method applicable to various variable 
# types. So, what's a non parametric method ?
#   Non-parametric method does not make explicit assumptions about functional
# form of f (any arbitary function). Instead, it tries to estimate f such that
# it can be as close to the data points without seeming impractical.
#   How does it work ? In simple words, it builds a random forest model for 
# each variable. Then it uses the model to predict missing values in the
# variable with the help of observed values.
#   It yield OOB (out of bag) imputation error estimate. Moreover, it 
# provides high level of control on imputation process. It has options to 
# return OOB separately (for each variable) instead of aggregating over the
# whole data matrix. This helps to look more closely as to how accurately 
# the model has imputed values for each variable.

#missForest
library(missForest)

#seed 10% missing values
iris.mis <- prodNA(iris, noNA = 0.1)
summary(iris.mis)

#impute missing values, using all parameters as default values
iris.imp <- missForest(iris.mis)

#check imputed values
iris.imp$ximp

#check imputation error
iris.imp$OOBerror

#    NRMSE         PFC 
# 0.14399312 0.04580153

#   NRMSE is normalized mean squared error. It is used to represent error 
# derived from imputing continuous values. 
#   PFC (proportion of falsely classified) is used to represent error 
# derived from imputing categorical values.

#comparing actual data accuracy
iris.err <- mixError(iris.imp$ximp, iris.mis, iris)
iris.err

#    NRMSE      PFC 
# 0.14755771 0.05263158 

# This suggests that categorical variables are imputed with 6% error
# and continuous variables are imputed with 15% error.
# This can be improved by tuning the values of mtry and ntree parameter.
# mtry refers to the number of variables being randomly sampled at each split. 
# ntree refers to number of trees to grow in the forest.


                ###################### Hmisc #########################
                ######################################################

##       Hmisc is a multiple purpose package useful for data analysis, high - level
# graphics, imputing missing values, advanced table making, model fitting &    
# diagnostics (linear regression, logistic regression & cox regression) etc.
# Amidst, the wide range of functions contained in this package, it offers 2 
# powerful functions for imputing missing values. These are impute() and 
# aregImpute(). Though, it also has transcan() function, but aregImpute() 
# is better to use.

##   impute() function simply imputes missing value using user defined statistical
# method (mean, max, mean). It's default is median. On the other hand, 
# aregImpute() allows mean imputation using additive regression, bootstrapping,
# and predictive mean matching.

##    In bootstrapping, different bootstrap resamples are used for each of multiple
# imputations. Then, a flexible additive model (non parametric regression method)
# is fitted on samples taken with replacements from original data and missing 
# values (acts as dependent variable) are predicted using non-missing values 
# (independent variable).

##     Then, it uses predictive mean matching (default) to impute missing values.
# Predictive mean matching works well for continuous and categorical (binary 
# & multi-level) without the need for computing residuals and maximum likelihood 
# fit.


##   Here are some important highlights of this package:
# It assumes linearity in the variables being predicted.
# Fisher's optimum scoring method is used for predicting categorical variables.

library(Hmisc)

#seed missing values ( 10% )
iris.mis <- prodNA(iris, noNA = 0.1)
summary(iris.mis)

# impute with mean value
iris.mis$imputed_age <- with(iris.mis, impute(Sepal.Length, mean))

# impute with random value
iris.mis$imputed_age2 <- with(iris.mis, impute(Sepal.Length, 'random'))

#similarly you can use min, max, median to impute missing value

#using argImpute
# argImpute() automatically identifies the variable type and treats them accordingly.
## The output shows R� values for predicted missing values. Higher the value, better
# are the values predicted.
impute_arg <- aregImpute(~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width +
                             Species, data = iris.mis, n.impute = 5)
impute_arg

#You can also check imputed values using the following command

#check imputed variable Sepal.Length
impute_arg$imputed$Sepal.Length



                ######################### mi #########################
                ######################################################

##    mi (Multiple imputation with diagnostics) package provides several features for 
# dealing with missing values. Like other packages, it also builds multiple imputation 
# models to approximate missing values. And, uses predictive mean matching method.

## Though, I've already explained predictive mean matching (pmm) above, but if you haven't 
# understood yet, here's a simpler version: For each observation in a variable with missing
# value, we find observation (from available values)  with the closest predictive mean to 
# that variable. The observed value from this "match" is then used as imputed value.

### >>>   Below are some unique characteristics of this package:
  
#    It allows graphical diagnostics of imputation models and convergence of imputation 
# process.
#    It uses bayesian version of regression models to handle issue of separation.
#    Imputation model specification is similar to regression output in R
#    It automatically detects irregularities in data such as high collinearity among variables.
#    Also, it adds noise to imputation process to solve the problem of additive constraints.

library(mi)

#seed missing values ( 10% )
iris.mis <- prodNA(iris, noNA = 0.1)
summary(iris.mis)

#imputing missing value with mi
mi_data <- mi(iris.mis, seed = 335)

#####  I've used default values of parameters namely:
# rand.imp.method as "bootstrap"
# n.imp (number of multiple imputations) as 3
# n.iter ( number of iterations) as 30

summary(mi_data)

##As shown, it uses summary statistics to define the imputed values.





