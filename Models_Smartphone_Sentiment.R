# Title:  Assn 4 Task 3 Sentiment Analysis for Smart Phones

# Last update: 2019.05.01

# File/project name: Models_Smartphone_Sentiment.R

# RStudio Project name: See resources for details on R projects

###############
# Project Notes
###############

# Project summary:  This is an R pipeline that will help us to perform 
# classification so that we can predict which brand of products customers prefer.

# This analysis found that random forest with no features removed was the most accurate model.
# When this model was used to predict which brand of computer would be preferred by customer's 
# who sent in incomplete survey responses, the result was Sony.

#########################   FILL THIS IN 
# Assignment "<-" short-cut: 
#   OSX [Alt]+[-] (next to "+" sign)
#   Win [Alt]+[-] 

###############
# Housekeeping
###############

# get working directory
getwd()
# set working directory
setwd("/Users/celestehofer/Desktop/Austin_DS_Program/Assn4_Task3")
dir()

################
# Load packages
################

install.packages("doMC") # specific to OSX
install.packages("corrplot")

install.packages("ggplot2")
install.packages("ggfortify")
install.packages("plotly")
install.packages("TTR")



require(doParallel)
require(foreach)
require(caret)
require(MASS)
require(mlbench)
require(readr)
require(doMC)
require(plyr)
require(corrplot)
require(rsample)
require(lubridate) 
require(ggplot2)
require(ggfortify)
require(forecast)
require(TTR)
require(plotly)
require(dplyr) 
require(tidyr)
require(tidyverse)

#####################
# Save image
#####################
save.image()

#####################
# Parallel processing
#####################

detectCores()   # detect number of cores, my OSX has 8
registerDoMC(cores = 4)  # set number of cores (don't use all available)


###############
# Import data
###############


## Load iPhone data (Dataset 1)
ds_complete <- read.csv("iphone_smallmatrix_labeled_8d.csv")
class(ds_complete)  # "data.frame"

str(ds_complete)


## Load LargeMatrix data to be used for predicting sentiment toward iphone
iphoneLargeMatrix <- read.csv("iphoneLargeMatrix.csv")
class(iphoneLargeMatrix)  # "data.frame"
str(iphoneLargeMatrix)


## Load Samsung samsung data (Dataset 2) ---#
ds_complete2 <- read.csv("galaxy_smallmatrix_labeled_9d.csv")
class(ds_complete2)  # "data.frame"
str(ds_complete2)

## Load LargeMatrix data to be used for predicting sentiment toward galaxy
galaxyLargeMatrix <- read.csv("galaxyLargeMatrix.csv")
class(galaxyLargeMatrix)  # "data.frame"
str(galaxyLargeMatrix)

################
# Evaluate data
################

#--- Dataset 1 ds_complete iphone---#

attributes(ds_complete) # Gives the attributes, class (data frame) and row names
str(ds_complete) # tells  num objs , variable types, and range of values
names(ds_complete) # gives attribute names
summary(ds_complete) #checking for count of NAs  Gives min, max, mean, median, quartile per attribute

# gives the type of each column
head(ds_complete)
tail(ds_complete)

# plot 
hist(ds_complete$ios)
hist(ds_complete$iphonecampos)
hist(ds_complete$samsungdisneg)  # replaces samsungdisneg
hist(ds_complete$htcperunc)
plot(ds_complete$ios, ds_complete$iosperneg)
qqnorm(ds_complete$ios)
plot_ly(ds_complete, x=ds_complete$iphonesentiment, type='histogram')
plot_ly(ds_complete, x=ds_complete$iphonesentiment, type='histogram')%>%
  layout(title = 'iPhone Sentiment Small Matrix', 
         xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
         yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))

# check for missing values 
is.na(ds_complete) # shows TRUE or FALSE on na
anyNA(ds_complete)
sum(is.na(ds_complete))  
# 0  no missing data

# create iphoneDF
iphoneDF <- ds_complete

#--- Dataset 2 ds_complete2 samsung samsung ---#

attributes(ds_complete2) # Gives the attributes, class (data frame) and row names
str(ds_complete2) # tells  num objs , variable types, and range of values
names(ds_complete2) # gives attribute names
summary(ds_complete2) #checking for count of NAs  Gives min, max, mean, median, quartile per attribute
# gives the type of each column
head(ds_complete2)
tail(ds_complete2)

# plot 
hist(ds_complete2$googleandroid)
hist(ds_complete2$samsungcampos)
hist(ds_complete2$samsungdisneg)  # replaces samsungdisneg
hist(ds_complete2$htcperunc)
plot(ds_complete2$googleandroid, ds_complete2$samsungperneg)
qqnorm(ds_complete2$googleandroid)
plot_ly(ds_complete2, x=ds_complete2$galaxysentiment, type='histogram')
plot_ly(ds_complete2, x=ds_complete2$galaxysentiment, type='histogram')%>%
  layout(title = 'Galaxy Sentiment Small Matrix', 
         xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
         yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))

# check for na values
is.na(ds_complete2) # shows TRUE or FALSE on na
anyNA(ds_complete2)
sum(is.na(ds_complete2))
# 0 NAs

# create samsung data frame
samsungDF <- ds_complete2

##########################
# Feature Selection (FS)
##########################

# Three primary methods
# 1. Filtering
# 2. Wrapper methods (e.g., RFE caret)
# 3. Embedded methods (e.g., varImp)

#####################
# Filtering
# Determine highly 
# correlated features
#####################

#--- Dataset 1 ds_complete iphone remove highly correlated features ---#
# good for num/int data 
iphoneCOR <-  iphoneDF
# calculate correlation matrix for all vars
corrAll <- cor(iphoneCOR[,1:59])
# summarize the correlation matrix
corrAll

# plot correlation matrix
corrplot(corrAll, order = "hclust") # sorts based on level of collinearity
corrplot(corrAll, method = "circle")

# find IVs that are highly correlated

correlationMatrix <- cor(iphoneCOR[,1:58])
print(correlationMatrix)
#options(max.print=1000000)

# 0.90 is the value we used earlier in our assignmments
highlyCorrIncPred <- findCorrelation(corrAll, verbose=TRUE, names=TRUE, cutoff=0.90, exact = TRUE) #exact = ncol(correlationMatrix) < 100)
print(highlyCorrIncPred)
# [1] "samsungdisneg" "samsungdispos" "googleperneg"  "samsungdisunc" "nokiacamunc"   "nokiadisneg"  
# [7] "nokiaperunc"   "nokiaperneg"   "nokiacamneg"   "iosperunc"     "iosperneg"     "ios"          
# [13] "htcphone" 

highlyCorrInputsOnly <- findCorrelation(correlationMatrix, cutoff=0.90, verbose = FALSE, names = TRUE, exact = TRUE)
print(highlyCorrInputsOnly)
# [1] "samsungdisneg" "samsungdispos" "googleperneg"  "samsungdisunc" "nokiacamunc"   "nokiadisneg"  
# [7] "nokiaperunc"   "nokiaperneg"   "nokiacamneg"   "iosperunc"     "iosperneg"     "ios"          
# [13] "htcphone"   

iphoneCOR$samsungdisneg <- NULL
iphoneCOR$samsungdispos <- NULL
iphoneCOR$googleperneg <- NULL
iphoneCOR$samsungdisunc <- NULL
iphoneCOR$nokiacamunc <- NULL
iphoneCOR$nokiadisneg <- NULL
iphoneCOR$nokiaperunc <- NULL
iphoneCOR$nokiaperneg <- NULL
iphoneCOR$nokiacamneg <- NULL
iphoneCOR$iosperunc <- NULL
iphoneCOR$iosperneg <- NULL
iphoneCOR$ios <- NULL
iphoneCOR$htcphone <- NULL

attributes(iphoneCOR)


#--- Dataset 2 ds_complete2 samsung galaxy remove highly correlated features---#
 
samsungCOR <-  ds_complete2
str(samsungCOR)
# calculate correlation matrix for all vars
corrAllsam <- cor(samsungCOR[,1:59])
# summarize the correlation matrix
corrAllsam

# plot correlation matrix
corrplot(corrAllsam, order = "hclust") # sorts based on level of collinearity
corrplot(corrAllsam, method = "circle")

# find IVs that are highly correlated

correlationMatrix2 <- cor(samsungCOR[,1:58])
print(correlationMatrix2)
#options(max.print=1000000)

# 0.90 is the value we used earlier in our assignmments
highlyCorrIncPred2 <- findCorrelation(corrAllsam, verbose=TRUE, names=TRUE, cutoff=0.90, exact = TRUE) #exact = ncol(correlationMatrix) < 100)
print(highlyCorrIncPred2)


highlyCorrInputsOnly2 <- findCorrelation(correlationMatrix2, cutoff=0.90, verbose = FALSE, names = TRUE, exact = TRUE)
print(highlyCorrInputsOnly2)

# [1] "samsungdisneg" "samsungdispos" "googleperneg"  "samsungdisunc" "nokiacamunc"   "nokiadisneg"   "nokiaperunc"  
# [8] "nokiaperneg"   "nokiacamneg"   "iosperunc"     "iosperneg"     "sonydisneg"    "ios"           "htcphone"   

samsungCOR$samsungdisneg <- NULL
samsungCOR$samsungdispos <- NULL
samsungCOR$googleperneg <- NULL
samsungCOR$samsungdisunc <- NULL
samsungCOR$nokiacamunc <- NULL
samsungCOR$nokiadisneg <- NULL
samsungCOR$nokiaperunc <- NULL
samsungCOR$nokiaperneg <- NULL
samsungCOR$nokiacamneg <- NULL
samsungCOR$iosperunc <- NULL
samsungCOR$iosperneg <- NULL
samsungCOR$sonydisneg <-  NULL
samsungCOR$ios <- NULL
samsungCOR$htcphone <- NULL

attributes(samsungCOR)
str(samsungCOR)
#########################
# Feature removal, manual
#########################

#--- Dataset 1 ---#


######################
# Feature removal, 
# automated, caret RFE
######################

# lmFuncs - linear model
# rfFuncs - random forests
# nbFuncs - naive Bayes
# treebagFuncs - bagged trees

## ---- rf iphone caret RFE ---- ##
set.seed(123)
# sample the data
iphoneSample <- iphoneDF[sample(1:nrow(iphoneDF), 1000, replace=FALSE),]

# define refControl using random forest
rfecontrol <- rfeControl(functions=rfFuncs, method="repeatedcv", number=5,verbose = FALSE)
# run the RFE algorithm

rfeResults <- rfe(iphoneSample[,1:58], iphoneSample$iphonesentiment, sizes=c(1:58), rfeControl=rfecontrol)

# rfeResults <- rfe(iphoneSample[,1:58], iphoneSample[,59], sizes=c(1:58), rfeControl=rfecontrol)
rfeResults
# Recursive feature selection

# Outer resampling method: Cross-Validated (5 fold, repeated 1 times) 

# Resampling performance over subset size:
  
#   Variables  RMSE Rsquared    MAE  RMSESD RsquaredSD   MAESD Selected
#           1 1.511   0.3402 1.1555 0.12019    0.09718 0.08017         
#           2 1.477   0.3792 1.1767 0.13634    0.12134 0.07890         
#           3 1.468   0.4000 1.1807 0.12017    0.10317 0.06742         
#           4 1.484   0.3978 1.2006 0.10721    0.10564 0.05930         
#           5 1.490   0.3981 1.1977 0.09975    0.10236 0.07023         
#           6 1.401   0.4331 1.0433 0.15797    0.12424 0.09389         
#           7 1.394   0.4391 1.0359 0.14714    0.11962 0.08169         
#           8 1.398   0.4359 1.0425 0.14759    0.12281 0.07563         
#           9 1.400   0.4331 1.0108 0.14416    0.11928 0.07979         
#         10 1.398   0.4344 1.0103 0.14670    0.12034 0.07716         
#         11 1.387   0.4430 1.0052 0.14584    0.11893 0.07944        *
#         12 1.394   0.4377 0.9901 0.14722    0.12113 0.08266         
#         13 1.394   0.4376 0.9948 0.14784    0.12172 0.08603         
#         14 1.393   0.4382 0.9969 0.14759    0.12188 0.08544         
#         15 1.398   0.4353 0.9841 0.14464    0.12014 0.08248         
#         16 1.398   0.4346 0.9837 0.14644    0.12113 0.08316         
#         17 1.394   0.4375 0.9851 0.14784    0.12142 0.08018         
#         18 1.397   0.4357 0.9775 0.14796    0.12112 0.08104         
#         19 1.398   0.4348 0.9807 0.14975    0.12260 0.07946         
#         20 1.399   0.4342 0.9849 0.14912    0.12293 0.07884         
#         21 1.401   0.4332 0.9783 0.14946    0.12270 0.07917         
#         22 1.399   0.4342 0.9808 0.15038    0.12290 0.07889    
#         23 1.398   0.4347 0.9817 0.15030    0.12297 0.07919         
#         24 1.401   0.4330 0.9783 0.15038    0.12305 0.08055         
#         25 1.399   0.4345 0.9779 0.14961    0.12290 0.07750         
#         26 1.399   0.4342 0.9790 0.15022    0.12292 0.08314         
#         27 1.396   0.4367 0.9724 0.14801    0.11980 0.08212         
#         28 1.396   0.4367 0.9743 0.14793    0.11935 0.08386         
#         29 1.396   0.4369 0.9772 0.15026    0.12155 0.08386         
#         30 1.398   0.4352 0.9740 0.14642    0.11912 0.08273         
#         31 1.397   0.4360 0.9739 0.14609    0.11867 0.08280         
#         32 1.395   0.4375 0.9743 0.14934    0.12105 0.08348         
#         33 1.397   0.4364 0.9709 0.15045    0.12158 0.08558         
#         34 1.395   0.4374 0.9735 0.15514    0.12471 0.08556         
#         35 1.395   0.4375 0.9737 0.15434    0.12343 0.08620         
#         36 1.395   0.4375 0.9719 0.15129    0.12199 0.08474         
#         37 1.396   0.4371 0.9738 0.15173    0.12258 0.08411         
#         38 1.396   0.4369 0.9740 0.15331    0.12284 0.08490         
#         39 1.398   0.4353 0.9732 0.15019    0.12123 0.08344         
#         40 1.395   0.4377 0.9733 0.15353    0.12348 0.08441         
#         41 1.395   0.4368 0.9752 0.15566    0.12471 0.08556         
#         42 1.397   0.4363 0.9724 0.15167    0.12251 0.08347         
#         43 1.396   0.4365 0.9729 0.15406    0.12335 0.08633         
#         44 1.393   0.4385 0.9729 0.15085    0.12160 0.08190         
#         45 1.396   0.4369 0.9725 0.15273    0.12181 0.08493  
#         46 1.394   0.4380 0.9711 0.15289    0.12253 0.08398         
#         47 1.394   0.4383 0.9717 0.15331    0.12292 0.08619         
#         48 1.398   0.4353 0.9739 0.15172    0.12276 0.08483         
#         49 1.396   0.4365 0.9730 0.15221    0.12299 0.08447         
#         50 1.397   0.4361 0.9753 0.15153    0.12249 0.08429         
#         51 1.397   0.4361 0.9731 0.14987    0.12104 0.08256         
#         52 1.396   0.4368 0.9717 0.15228    0.12253 0.08530         
#         53 1.395   0.4375 0.9733 0.15141    0.12161 0.08162         
#         54 1.396   0.4370 0.9719 0.14967    0.12031 0.08348         
#         55 1.396   0.4366 0.9736 0.15072    0.12158 0.08305         
#         56 1.395   0.4370 0.9738 0.15175    0.12201 0.08327         
#         57 1.397   0.4363 0.9695 0.15202    0.12210 0.08451         
#         58 1.394   0.4383 0.9705 0.15250    0.12262 0.08483         

# The top 5 variables (out of 11):
#  iphone, googleandroid, samsunggalaxy, iphonedispos, iphonedisunc


plot(rfeResults, type=c("g", "o")) 

# show predictors used
predictors(rfeResults) 
# [1] "iphone"        "googleandroid" "samsunggalaxy" "iphonedisunc"  "iphoneperunc" 
# [6] "sonyxperia"    "iphonecamunc"  "htcphone"      "iphonecampos"  "iphonedispos" 
# [11] "iphoneperneg"  "iphoneperpos"  "iphonecamneg"  "iphonedisneg"  "htccampos"    
# [16] "htcdisneg"     "htccamneg"     "ios"           "htcdispos"     "samsungperpos"
# [21] "samsungperunc"

varImp(rfeResults)

#                 Overall
# iphone        73.187797
# googleandroid 29.344559
# samsunggalaxy 24.842981
# iphonedisunc  20.938169
# iphoneperunc  18.555416
# sonyxperia    16.979054
# iphonecamunc  16.754077
# htcphone      16.667520
# iphonedispos  14.845529
# iphonecampos  14.556917
# iphoneperneg  14.466079
# iphoneperpos  12.552935
# iphonecamneg  12.247082
# iphonedisneg  11.651768
# htccampos     10.855830
# htcdisneg     10.622315
# ios            9.341933
# htccamneg      8.844488
# samsungperunc  7.932984
# htcdispos      7.457718
# samsungperpos  7.096678
# samsungcamunc  6.319742
# samsungperneg  5.838311
# samsungcamneg  5.747824

# create new ds with only the rfe recommended features
iphoneRFE <- iphoneDF[,predictors(rfeResults)]  # only takes the top 21 of 23 variables

# add target or dependent variable back in
iphoneRFE$iphonesentiment <-  iphoneDF$iphonesentiment


str(iphoneRFE)  
# 'data.frame':	12973 obs. of  22 variables:
# $ iphone         : int  1 1 1 1 1 41 1 1 1 1 ...
  # $ googleandroid  : int  0 0 0 0 0 0 0 0 0 0 ...
  # $ samsunggalaxy  : int  0 0 0 0 0 0 0 0 0 0 ...
  # $ iphonedisunc   : int  0 0 0 0 0 4 9 0 0 0 ...
  # $ iphoneperunc   : int  0 0 0 1 0 0 5 0 0 0 ...
  # $ sonyxperia     : int  0 0 0 0 0 0 0 0 0 0 ...
  # $ iphonecamunc   : int  0 0 0 0 0 7 1 0 0 0 ...
  # $ htcphone       : int  0 0 0 0 0 0 0 0 0 0 ...
  # $ iphonecampos   : int  0 0 0 0 0 1 1 0 0 0 ...
  # $ iphonedispos   : int  0 0 0 0 0 1 13 0 0 0 ...
  # $ iphoneperneg   : int  0 0 0 0 0 0 4 1 0 0 ...
# $ iphoneperpos   : int  0 1 0 1 1 0 5 3 0 0 ...# 
# $ iphonecamneg   : int  0 0 0 0 0 3 1 0 0 0 ...
# $ iphonedisneg   : int  0 0 0 0 0 3 10 0 0 0 ...
# $ htccampos      : int  0 0 0 0 0 0 0 0 0 0 ...
# $ htcdisneg      : int  0 0 0 0 0 0 0 0 0 0 ...
# $ htccamneg      : int  0 0 0 0 0 0 0 0 0 0 ...
# $ ios            : int  0 0 0 0 0 6 0 0 0 0 ...
# $ htcdispos      : int  0 0 0 0 0 0 0 0 0 0 ...
# $ samsungperpos  : int  0 0 0 0 0 0 0 0 0 0 ...
# $ samsungperunc  : int  0 0 0 0 0 0 0 0 0 0 ...
# $ iphonesentiment: Factor w/ 6 levels "very_negative",..: 1 1 1 1 1 5 5 1 1 1 ...

## ---- rf galaxy caret RFE ---- ##
set.seed(123)
# sample the data
samsungSample <- ds_complete2[sample(1:nrow(ds_complete2), 1000, replace=FALSE),]
samsungSample
# define refControl using random forest
rfecontrol2 <- rfeControl(functions=rfFuncs, method="repeatedcv", number=5,verbose = FALSE)
# run the RFE algorithm
rfecontrol2
rfeResults2 <- rfe(samsungSample[,1:58], samsungSample$galaxysentiment, sizes=c(1:58), rfeControl=rfecontrol2)
rfeResults2
# Recursive feature selection

# Outer resampling method: Cross-Validated (5 fold, repeated 1 times) 

# Resampling performance over subset size:
  
#   Variables  RMSE Rsquared    MAE  RMSESD RsquaredSD   MAESD Selected
# 1 1.531   0.2954 1.1542 0.06136    0.06584 0.03601         
# 2 1.510   0.3170 1.1726 0.05076    0.06550 0.03419         
# 3 1.507   0.3312 1.1915 0.06621    0.08707 0.04531         
# 4 1.520   0.3294 1.2113 0.06116    0.08388 0.03733         
# 5 1.510   0.3484 1.2013 0.04900    0.08050 0.02764         
# 6 1.432   0.3852 1.0683 0.06959    0.06545 0.03278  
# 7 1.423   0.3943 1.0550 0.05934    0.05738 0.02798         
# 8 1.410   0.4047 1.0464 0.06061    0.05834 0.02907         
# 9 1.400   0.4106 0.9991 0.05103    0.05694 0.02916         
# 10 1.397   0.4133 0.9984 0.06204    0.06359 0.04247         
# 11 1.388   0.4207 0.9957 0.05955    0.06535 0.03830        *
# 12 1.396   0.4142 0.9849 0.05937    0.06537 0.04033         
# 13 1.400   0.4106 0.9895 0.06021    0.06684 0.04247         
# 14 1.401   0.4100 0.9928 0.06242    0.06864 0.04281         
# 15 1.405   0.4073 0.9865 0.06988    0.07431 0.04875         
# 16 1.405   0.4076 0.9873 0.06974    0.07596 0.04505         
# 17 1.405   0.4072 0.9918 0.07095    0.07790 0.04430         
# 18 1.410   0.4039 0.9903 0.06674    0.07344 0.04308         
# 19 1.412   0.4026 0.9915 0.06905    0.07579 0.04320         
# 20 1.407   0.4054 0.9919 0.06974    0.07596 0.04386         
# 21 1.413   0.4013 0.9918 0.06429    0.07121 0.04424         
# 22 1.413   0.4011 0.9932 0.06707    0.07325 0.04458  
# 23 1.409   0.4040 0.9923 0.06660    0.07364 0.04294         
# 24 1.412   0.4024 0.9904 0.06592    0.07259 0.04229         
# 25 1.411   0.4025 0.9919 0.06437    0.07252 0.04169         
# 26 1.410   0.4031 0.9921 0.06566    0.07433 0.04301         
# 27 1.412   0.4023 0.9921 0.06449    0.07313 0.04244         
# 28 1.411   0.4030 0.9926 0.06373    0.07167 0.04117         
# 29 1.413   0.4015 0.9936 0.06151    0.06942 0.04090         
# 30 1.412   0.4023 0.9917 0.06255    0.07125 0.04192         
# 31 1.413   0.4013 0.9921 0.06556    0.07325 0.04431         
# 32 1.411   0.4026 0.9936 0.06242    0.07013 0.04131         
# 33 1.415   0.3994 0.9949 0.06225    0.06941 0.04242         
# 34 1.410   0.4035 0.9930 0.06424    0.07129 0.04278         
# 35 1.411   0.4029 0.9941 0.06382    0.07166 0.04288         
# 36 1.411   0.4030 0.9907 0.06438    0.07100 0.04281         
# 37 1.410   0.4031 0.9909 0.06591    0.07263 0.04501         
# 38 1.409   0.4041 0.9926 0.06625    0.07327 0.04458  
# 39 1.411   0.4025 0.9921 0.06447    0.07214 0.04425         
# 40 1.408   0.4052 0.9912 0.06386    0.07148 0.04269         
# 41 1.409   0.4043 0.9926 0.06465    0.07249 0.04402         
# 42 1.412   0.4019 0.9922 0.06462    0.07224 0.04342         
# 43 1.409   0.4042 0.9915 0.06841    0.07532 0.04689         
# 44 1.409   0.4040 0.9924 0.06598    0.07323 0.04586         
# 45 1.410   0.4035 0.9914 0.06544    0.07324 0.04575         
# 46 1.411   0.4028 0.9916 0.06708    0.07375 0.04423         
# 47 1.408   0.4048 0.9912 0.06605    0.07324 0.04461         
# 48 1.410   0.4037 0.9910 0.06610    0.07334 0.04518         
# 49 1.411   0.4027 0.9917 0.06623    0.07254 0.04443         
# 50 1.409   0.4043 0.9935 0.06760    0.07515 0.04501         
# 51 1.410   0.4035 0.9906 0.06581    0.07338 0.04446         
# 52 1.410   0.4035 0.9927 0.06616    0.07418 0.04575         
# 53 1.407   0.4059 0.9885 0.06817    0.07492 0.04619         
# 54 1.409   0.4041 0.9888 0.06606    0.07206 0.04511         
# 55 1.406   0.4072 0.9868 0.06644    0.07302 0.04231         
# 56 1.408   0.4053 0.9906 0.06473    0.07163 0.04102         
# 57 1.410   0.4038 0.9900 0.06414    0.07087 0.04155         
# 58 1.410   0.4038 0.9902 0.06483    0.07262 0.04425 

# The top 5 variables (out of 11):
#   iphone, iphonedispos, iphonedisunc, googleandroid, iphonecampos

plot(rfeResults2, type=c("g", "o")) 

# show predictors used
predictors(rfeResults2) 
# [1] "iphone"        "iphonedispos"  "iphonedisunc"  "googleandroid" "iphonecampos"  "samsunggalaxy" "sonyxperia"   
# [8] "iphoneperpos"  "htcphone"      "htccampos"     "iphonecamneg" 
varImp(rfeResults2)

# create new ds with only the rfe recommended features
samsungRFE <- ds_complete2[,predictors(rfeResults2)]  # only takes the top 21 of 23 variables

# add target or dependent variable back in
samsungRFE$galaxysentiment <-  ds_complete2$galaxysentiment


str(samsungRFE) 

# 'data.frame':	12911 obs. of  12 variables:
# $ iphone         : int  1 1 1 0 1 2 1 1 4 1 ...
# $ iphonedispos   : int  0 1 0 0 0 0 2 0 0 0 ...
# $ iphonedisunc   : int  0 1 0 0 0 0 0 0 0 0 ...
# $ googleandroid  : int  0 0 0 0 0 0 0 0 0 0 ...
# $ iphonecampos   : int  0 0 1 0 0 1 0 0 0 0 ...
# $ samsunggalaxy  : int  0 0 1 0 0 0 0 0 0 0 ...
# $ sonyxperia     : int  0 0 0 0 0 0 0 0 0 0 ...
# $ iphoneperpos   : int  0 0 0 0 0 0 0 0 0 0 ...
# $ htcphone       : int  0 0 0 1 0 0 0 0 0 0 ...
# $ htccampos      : int  0 0 0 0 0 0 0 0 0 0 ...
# $ iphonecamneg   : int  0 0 0 0 0 0 0 0 0 0 ...
# $ galaxysentiment: int  5 3 3 0 1 0 3 5 5 5 ...

#############
# Preprocess
#############

#--- Dataset 1 iphone---#



iphoneDF$iphonesentiment <- as.factor(iphoneDF$iphonesentiment)

dFramelist <- list(iphoneDF, iphoneCOR, iphoneNZV, iphoneRFE )
dFramelist <- list(iphoneDF)
dFrameListTarget <- lapply(dFramelist, '[[', 'iphonesentiment')
dFrameListTarget
str(dFramelist)
# Change 0 through 5 to their corresponding values in sentiment
for(i in dFramelist){print(i)}
dFramelist[[1]]#$iphonesentiment
#for(i in 1:dFramelist)
  for(i in dFramelistTarget)
{dFrameListTarget[i] <- revalue(dFrameListTarget[i], c("0"="very_negative"))
dFramelist[[i]] <- revalue(dFramelist[[i]], c("1"="negative"))
dFramelist[[i]] <- revalue(dFramelist[[i]], c("2"="somewhat_negative"))
dFramelist[[i]] <- revalue(dFramelist[[i]], c("3"="somewhat_positive"))
dFramelist[[i]] <- revalue(dFramelist[[i]], c("4"="positive"))
dFramelist[[i]] <- revalue(dFramelist[[i]], c("5"="very_positive"))
}


#--- Dataset 1 iphone near zero var---#
# Near Zero Var
# This needed to be implemented to remove predictors/inputs with zero variance
# Returns a table with: frequency ratio, percentage unique, zero variance and near zero variance
nzvMetrics <- nearZeroVar(iphoneDF,saveMetrics = TRUE)
nzvMetrics

nzv <- nearZeroVar(iphoneDF,
                   saveMetrics = FALSE)
nzv
# [1]  3  4  6  7  9 10 11 12 13 14 15 16 17 19 20 21 22 24 25 26 27 29 30 31 32 34 35 36 37 39 40 41 42 44 45 46 47
# [38] 49 50 51 52 53 54 55 56 57 58

# The ds has 59 attributes.  According to nzv47 are near Zero Variance
iphoneNZV <- iphoneDF[,-nzv]
str(iphoneNZV)

# 'data.frame':	12973 obs. of  12 variables:
#   $ iphone         : int  1 1 1 1 1 41 1 1 1 1 ...
# $ samsunggalaxy  : int  0 0 0 0 0 0 0 0 0 0 ...
# $ htcphone       : int  0 0 0 0 0 0 0 0 0 0 ...
# $ iphonecampos   : int  0 0 0 0 0 1 1 0 0 0 ...
# $ iphonecamunc   : int  0 0 0 0 0 7 1 0 0 0 ...
# $ iphonedispos   : int  0 0 0 0 0 1 13 0 0 0 ...
# $ iphonedisneg   : int  0 0 0 0 0 3 10 0 0 0 ...
# $ iphonedisunc   : int  0 0 0 0 0 4 9 0 0 0 ...
# $ iphoneperpos   : int  0 1 0 1 1 0 5 3 0 0 ...
# $ iphoneperneg   : int  0 0 0 0 0 0 4 1 0 0 ...
# $ iphoneperunc   : int  0 0 0 1 0 0 5 0 0 0 ...
# $ iphonesentiment: int  0 0 0 0 0 4 4 0 0 0 ...

dim(iphoneNZV)
# [1] 12973    12

#--- Dataset 2 samsung galaxy phone---#

# Near Zero Var
# This needed to be implemented to remove predictors/inputs with zero variance
# Returns a table with: frequency ratio, percentage unique, zero variance and near zero variance
nzv2 <- nearZeroVar(ds_complete2,
                   saveMetrics = FALSE)
nzv2
count_nzvTrue <- sum(nzv2$nzv)
count_nzvTrue  
# 47

# The ds has 59 attributes.  According to nzv47 are near Zero Variance
samsungNZV <- ds_complete2[,-nzv2]
str(samsungNZV)

# 'data.frame':	12911 obs. of  12 variables:
#   $ iphone         : int  1 1 1 0 1 2 1 1 4 1 ...
# $ samsunggalaxy  : int  0 0 1 0 0 0 0 0 0 0 ...
# $ htcphone       : int  0 0 0 1 0 0 0 0 0 0 ...
# $ iphonecampos   : int  0 0 1 0 0 1 0 0 0 0 ...
# $ iphonecamunc   : int  0 0 0 0 0 0 0 0 0 0 ...
# $ iphonedispos   : int  0 1 0 0 0 0 2 0 0 0 ...
# $ iphonedisneg   : int  0 1 0 0 0 0 0 0 0 0 ...
# $ iphonedisunc   : int  0 1 0 0 0 0 0 0 0 0 ...
# $ iphoneperpos   : int  0 0 0 0 0 0 0 0 0 0 ...
# $ iphoneperneg   : int  0 0 0 0 0 0 0 0 0 0 ...
# $ iphoneperunc   : int  0 0 0 0 0 0 0 0 0 0 ...
# $ galaxysentiment: int  5 3 3 0 1 0 3 5 5 5 ...
dim(samsungNZV)
# [1] 12911    12

#--- Dataset 1 iphone revalue target values ---#
# change here for both iphonesentiment
iphoneDF$iphonesentiment <- as.factor(iphoneDF$iphonesentiment)

iphoneDF$iphonesentiment <- revalue(iphoneDF$iphonesentiment, c("0"="very_negative"))
iphoneDF$iphonesentiment <- revalue(iphoneDF$iphonesentiment, c("1"="negative"))
iphoneDF$iphonesentiment <- revalue(iphoneDF$iphonesentiment, c("2"="somewhat_negative"))
iphoneDF$iphonesentiment <- revalue(iphoneDF$iphonesentiment, c("3"="somewhat_positive"))
iphoneDF$iphonesentiment <- revalue(iphoneDF$iphonesentiment, c("4"="positive"))
iphoneDF$iphonesentiment <- revalue(iphoneDF$iphonesentiment, c("5"="very_positive"))

# change here for both iphonesentiment
iphoneNZV$iphonesentiment <- as.factor(iphoneNZV$iphonesentiment)

iphoneNZV$iphonesentiment <- revalue(iphoneNZV$iphonesentiment, c("0"="very_negative"))
iphoneNZV$iphonesentiment <- revalue(iphoneNZV$iphonesentiment, c("1"="negative"))
iphoneNZV$iphonesentiment <- revalue(iphoneNZV$iphonesentiment, c("2"="somewhat_negative"))
iphoneNZV$iphonesentiment <- revalue(iphoneNZV$iphonesentiment, c("3"="somewhat_positive"))
iphoneNZV$iphonesentiment <- revalue(iphoneNZV$iphonesentiment, c("4"="positive"))
iphoneNZV$iphonesentiment <- revalue(iphoneNZV$iphonesentiment, c("5"="very_positive"))

iphoneRFE$iphonesentiment <- as.factor(iphoneRFE$iphonesentiment)

iphoneRFE$iphonesentiment <- revalue(iphoneRFE$iphonesentiment, c("0"="very_negative"))
iphoneRFE$iphonesentiment <- revalue(iphoneRFE$iphonesentiment, c("1"="negative"))
iphoneRFE$iphonesentiment <- revalue(iphoneRFE$iphonesentiment, c("2"="somewhat_negative"))
iphoneRFE$iphonesentiment <- revalue(iphoneRFE$iphonesentiment, c("3"="somewhat_positive"))
iphoneRFE$iphonesentiment <- revalue(iphoneRFE$iphonesentiment, c("4"="positive"))
iphoneRFE$iphonesentiment <- revalue(iphoneRFE$iphonesentiment, c("5"="very_positive"))

iphoneCOR$iphonesentiment <- as.factor(iphoneCOR$iphonesentiment)

iphoneCOR$iphonesentiment <- revalue(iphoneCOR$iphonesentiment, c("0"="very_negative"))
iphoneCOR$iphonesentiment <- revalue(iphoneCOR$iphonesentiment, c("1"="negative"))
iphoneCOR$iphonesentiment <- revalue(iphoneCOR$iphonesentiment, c("2"="somewhat_negative"))
iphoneCOR$iphonesentiment <- revalue(iphoneCOR$iphonesentiment, c("3"="somewhat_positive"))
iphoneCOR$iphonesentiment <- revalue(iphoneCOR$iphonesentiment, c("4"="positive"))
iphoneCOR$iphonesentiment <- revalue(iphoneCOR$iphonesentiment, c("5"="very_positive"))

str(iphoneNZV)
str(iphoneCOR)
str(iphoneRFE)
str(iphoneDF)
head(iphoneNZV)
tail(iphoneCOR)
tail(iphoneDF)
head(iphoneRFE)

#--- Dataset 2 samsung revalue target values ---#
# change here for both galaxysentiment
samsungDF$galaxysentiment <- as.factor(samsungDF$galaxysentiment)

samsungDF$galaxysentiment <- revalue(samsungDF$galaxysentiment, c("0"="very_negative"))
samsungDF$galaxysentiment <- revalue(samsungDF$galaxysentiment, c("1"="negative"))
samsungDF$galaxysentiment <- revalue(samsungDF$galaxysentiment, c("2"="somewhat_negative"))
samsungDF$galaxysentiment <- revalue(samsungDF$galaxysentiment, c("3"="somewhat_positive"))
samsungDF$galaxysentiment <- revalue(samsungDF$galaxysentiment, c("4"="positive"))
samsungDF$galaxysentiment <- revalue(samsungDF$galaxysentiment, c("5"="very_positive"))

# change here for both galaxysentiment
samsungNZV$galaxysentiment <- as.factor(samsungNZV$galaxysentiment)

samsungNZV$galaxysentiment <- revalue(samsungNZV$galaxysentiment, c("0"="very_negative"))
samsungNZV$galaxysentiment <- revalue(samsungNZV$galaxysentiment, c("1"="negative"))
samsungNZV$galaxysentiment <- revalue(samsungNZV$galaxysentiment, c("2"="somewhat_negative"))
samsungNZV$galaxysentiment <- revalue(samsungNZV$galaxysentiment, c("3"="somewhat_positive"))
samsungNZV$galaxysentiment <- revalue(samsungNZV$galaxysentiment, c("4"="positive"))
samsungNZV$galaxysentiment <- revalue(samsungNZV$galaxysentiment, c("5"="very_positive"))

samsungRFE$galaxysentiment <- as.factor(samsungRFE$galaxysentiment)

samsungRFE$galaxysentiment <- revalue(samsungRFE$galaxysentiment, c("0"="very_negative"))
samsungRFE$galaxysentiment <- revalue(samsungRFE$galaxysentiment, c("1"="negative"))
samsungRFE$galaxysentiment <- revalue(samsungRFE$galaxysentiment, c("2"="somewhat_negative"))
samsungRFE$galaxysentiment <- revalue(samsungRFE$galaxysentiment, c("3"="somewhat_positive"))
samsungRFE$galaxysentiment <- revalue(samsungRFE$galaxysentiment, c("4"="positive"))
samsungRFE$galaxysentiment <- revalue(samsungRFE$galaxysentiment, c("5"="very_positive"))

samsungCOR$galaxysentiment <- as.factor(samsungCOR$galaxysentiment)

samsungCOR$galaxysentiment <- revalue(samsungCOR$galaxysentiment, c("0"="very_negative"))
samsungCOR$galaxysentiment <- revalue(samsungCOR$galaxysentiment, c("1"="negative"))
samsungCOR$galaxysentiment <- revalue(samsungCOR$galaxysentiment, c("2"="somewhat_negative"))
samsungCOR$galaxysentiment <- revalue(samsungCOR$galaxysentiment, c("3"="somewhat_positive"))
samsungCOR$galaxysentiment <- revalue(samsungCOR$galaxysentiment, c("4"="positive"))
samsungCOR$galaxysentiment <- revalue(samsungCOR$galaxysentiment, c("5"="very_positive"))

str(samsungNZV)
str(samsungCOR)
str(samsungRFE)
str(samsungDF)
head(samsungNZV)
tail(samsungCOR)
tail(samsungDF)
head(samsungRFE)



#--- Dataset iphoneLargeMatrix change target variable to factor type ---#

iphoneLargeMatrix$iphonesentiment <- as.factor(iphoneLargeMatrix$iphonesentiment)  
str(iphoneLargeMatrix)


#--- Dataset galaxyLargeMatrix change target variable to factor type ---#

galaxyLargeMatrix$galaxysentiment <- as.factor(galaxyLargeMatrix$galaxysentiment)  
str(galaxyLargeMatrix)


#######################
# FEATURE ENGINEERING
#######################

#--- Dataset 1 iphone re-engineer (recoding sentiment) target values ---#
# create a new dataset that will be used for recoding sentiment
iphoneRC <- ds_complete
# recode sentiment to combine factor levels 0 & 1 and 4 & 5
iphoneRC$iphonesentiment <- recode(iphoneRC$iphonesentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4) 
# inspect results
summary(iphoneRC)
str(iphoneRC)

# make iphonesentiment a factor
iphoneRC$iphonesentiment <- as.factor(iphoneRC$iphonesentiment)

iphoneRC$iphonesentiment <- revalue(iphoneRC$iphonesentiment, c("1"="negative"))
iphoneRC$iphonesentiment <- revalue(iphoneRC$iphonesentiment, c("2"="somewhat_negative"))
iphoneRC$iphonesentiment <- revalue(iphoneRC$iphonesentiment, c("3"="somewhat_positive"))
iphoneRC$iphonesentiment <- revalue(iphoneRC$iphonesentiment, c("4"="positive"))

#--- Dataset 1 iphone re-engineer Principal Component Analysis PCA ---#


# data = training and testing from iphoneDF (no feature selection) 
# create object containing centered, scaled PCA components from training set
# excluded the dependent variable and set threshold to .95

##### NOTE THIS CAN ONLY BE EXECUTED AFTER trainingALL  is created under the Train/Test sets section
preprocessParams <- preProcess(trainingALL[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParams)
# Created from 9083 samples and 58 variables

# Pre-processing:
#   - centered (58)
# - ignored (0)
# - principal component signal extraction (58)
# - scaled (58)

# PCA needed 25 components to capture 95 percent of the variance

# use predict to apply pca parameters, create training, exclude dependant
train.pca <- predict(preprocessParams, trainingALL[,-59])
str(train.pca)

# add in the target variable
train.pca$iphonesentiment <- trainingALL$iphonesentiment

# use predict to apply pca parameters, create testing, exclude dependant
test.pca <- predict(preprocessParams, testingALL[,-59])

# add the dependent to training
test.pca$iphonesentiment <- testingALL$iphonesentiment

# inspect results
str(train.pca)
head(train.pca)
str(test.pca)

#--- Dataset 1 iphone re-engineer Principal Component Analysis PCA and recoded target ---#

##### NOTE THIS CAN ONLY BE EXECUTED AFTER trainingRC  is created under the Train/Test sets section
preprocessParamsRC <- preProcess(trainingRC[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParamsRC)

# Created from 9083 samples and 58 variables

# Pre-processing:
#   - centered (58)
# - ignored (0)
# - principal component signal extraction (58)
# - scaled (58)

# PCA needed 27 components to capture 95 percent of the variance

# use predict to apply pca parameters, create training, exclude dependant
train.pcaRC <- predict(preprocessParamsRC, trainingRC[,-59])
str(train.pcaRC)

# add in the target variable to training
train.pcaRC$iphonesentiment <- trainingRC$iphonesentiment

# use predict to apply pca parameters, create testing, exclude dependant
test.pcaRC <- predict(preprocessParamsRC, testingRC[,-59])

# add the dependent to testing
test.pcaRC$iphonesentiment <- testingRC$iphonesentiment


# inspect results
str(train.pcaRC)
head(train.pcaRC)
str(test.pcaRC)

#--- Dataset 2 samsung re-engineer (recoding sentiment) target values ---#
# create a new dataset that will be used for recoding sentiment
samsungRC <- ds_complete2
# recode sentiment to combine factor levels 0 & 1 and 4 & 5
samsungRC$galaxysentiment <- recode(samsungRC$galaxysentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4) 
# inspect results
summary(samsungRC)
str(samsungRC)

# make iphonesentiment a factor
samsungRC$galaxysentiment <- as.factor(samsungRC$galaxysentiment)

samsungRC$galaxysentiment <- revalue(samsungRC$galaxysentiment, c("1"="negative"))
samsungRC$galaxysentiment <- revalue(samsungRC$galaxysentiment, c("2"="somewhat_negative"))
samsungRC$galaxysentiment <- revalue(samsungRC$galaxysentiment, c("3"="somewhat_positive"))
samsungRC$galaxysentiment <- revalue(samsungRC$galaxysentiment, c("4"="positive"))

str(samsungRC)
#--- Dataset 2 samsung re-engineer Principal Component Analysis PCA ---#


# data = training and testing from iphoneDF (no feature selection) 
# create object containing centered, scaled PCA components from training set
# excluded the dependent variable and set threshold to .95

##### NOTE THIS CAN ONLY BE EXECUTED AFTER trainingALL  is created under the Train/Test sets section
preprocessParams2 <- preProcess(trainingALLsam[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParams2)
#Created from 9040 samples and 58 variables

# Pre-processing:
#   - centered (58)
# - ignored (0)
# - principal component signal extraction (58)
# - scaled (58)

# PCA needed 25 components to capture 95 percent of the variance
# use predict to apply pca parameters, create training, exclude dependant
train.pcaSam <- predict(preprocessParams2, trainingALLsam[,-59])
str(train.pcaSam)

# add in the target variable
train.pcaSam$galaxysentiment <- trainingALLsam$galaxysentiment

# use predict to apply pca parameters, create testing, exclude dependant
test.pcaSam <- predict(preprocessParams2, testingALLsam[,-59])

# add the dependent to training
test.pcaSam$galaxysentiment <- testingALLsam$galaxysentiment

# inspect results
str(train.pcaSam)
head(train.pcaSam)
str(test.pcaSam)

#--- Dataset 2 samsung re-engineer Principal Component Analysis PCA and recoded target ---#

##### NOTE THIS CAN ONLY BE EXECUTED AFTER trainingRCSam  is created under the Train/Test sets section
preprocessParamsRCSam <- preProcess(trainingRCSam[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParamsRCSam)

# Created from 9039 samples and 58 variables

# Pre-processing:
#   - centered (58)
# - ignored (0)
# - principal component signal extraction (58)
# - scaled (58)

# PCA needed 24 components to capture 95 percent of the variance

# use predict to apply pca parameters, create training, exclude dependant
train.pcaRCSam <- predict(preprocessParamsRCSam, trainingRCSam[,-59])
str(train.pcaRCSam)


# 'data.frame':	9039 obs. of  24 variables:
#   $ PC1 : num  -0.17 0.104 -0.674 -0.622 -0.607 ...
# $ PC2 : num  -0.0604 -0.1933 0.0552 0.0457 0.0417 ...
# $ PC3 : num  -0.66042 -0.00748 -0.18229 -0.1161 -0.06014 ...
# $ PC4 : num  0.726 0.849 0.569 0.453 0.355 ...
# $ PC5 : num  -0.303 0.9394 -0.0308 0.0238 -0.0593 ...
# $ PC6 : num  -0.4461 0.852 0.1375 -0.0389 0.1942 ...
# $ PC7 : num  -0.0937 0.0617 -0.1636 -0.1204 -0.2796 ...
# $ PC8 : num  0.6536 -0.6893 -0.0908 -0.0631 -0.0362 ...
# $ PC9 : num  -0.2752 0.2236 0.0252 0.0187 -0.014 ...
# $ PC10: num  0.1663 -0.012 -0.0493 -0.0375 -0.0313 ...
# $ PC11: num  -0.79791 -0.16347 0.08972 0.05293 0.00396 ...
# $ PC12: num  -0.688 0.186 0.123 0.125 0.162 ...
# $ PC13: num  0.3867 0.0446 -0.0702 -0.4659 0.0127 ...
# $ PC14: num  2.311 -0.0667 -0.1795 -0.0485 -0.1777 ...
# $ PC15: num  0.112 -0.187 -0.125 -0.236 -0.287 ...
# $ PC16: num  -0.1099 0.4365 -0.0297 -0.0196 -0.0785 ...
# $ PC17: num  -0.00232 -0.38766 -0.02758 -0.05599 -0.08216 ...
# $ PC18: num  0.1344 0.5937 -0.0285 0.051 -0.024 ...
# $ PC19: num  -1.3622 -0.0665 0.0245 0.0235 0.0224 ...
# $ PC20: num  -0.04312 0.19763 0.02617 -0.00935 -0.02185 ...
# $ PC21: num  0.41439 -0.16686 -0.00502 0.02812 0.01544 ...
# $ PC22: num  -1.30545 0.05451 -0.00937 0.03779 0.01318 ...
# $ PC23: num  0.6385 0.2699 -0.0665 -0.0134 -0.0507 ...
# $ PC24: num  -0.1882 -0.1262 -0.0206 -0.0638 -0.0712 ...

# add in the target variable to training
train.pcaRCSam$galaxysentiment <- trainingRCSam$galaxysentiment

# use predict to apply pca parameters, create testing, exclude dependant
test.pcaRCSam <- predict(preprocessParamsRCSam, testingRCSam[,-59])

# add the dependent to testing
test.pcaRCSam$galaxysentiment <- testingRCSam$galaxysentiment

# inspect results
str(train.pcaRCSam)
head(train.pcaRCSam)
str(test.pcaRCSam)
##############################
# Variable Importance (varImp)
##############################

# varImp is evaluated in the model train/fit section


# ---- Conclusion ---- #

# RF is top model; compare selected predictors to varImp


##################
# Train/Test sets
##################

# First, we split the data into two groups: a training set and a test set
# createDataPartition does a stratified random split of the data.

set.seed(123)

#--- Dataset iphone , no features removed ---#
inTrainALL <- createDataPartition(y = iphoneDF$iphonesentiment, 
                                  ## ## the outcome data are needed
                                  p =.70, 
                                  ## ## The percentage of data in the
                                  ##  ## training set
                                  list = FALSE)
## The format of the results
## The output is a set of integers that belong in the training set.

# to partition the data into training and testing datasets
trainingALL <- iphoneDF[ inTrainALL,]
testingALL <- iphoneDF[-inTrainALL,]
nrow(trainingALL)
# verify number of obs 
str(trainingALL)  
str(testingALL) 


#--- Dataset correleted features removed ---#

inTrainCOR <- createDataPartition(y = iphoneCOR$iphonesentiment, 
                                  ## ## the outcome data are needed
                                  p =.70, 
                                  ## ## The percentage of data in the
                                  ##  ## training set
                                  list = FALSE)
## The format of the results
## The output is a set of integers that belong in the training set.

# to partition the data into training and testing datasets
trainingCOR <- iphoneCOR[ inTrainCOR,]
testingCOR <- iphoneCOR[-inTrainCOR,]
nrow(trainingCOR)
nrow(testingCOR)
# verify number of obs 
str(trainingCOR)  
str(testingCOR) 

#--- Dataset near zero variance features removed iphone---#

inTrainNZV <- createDataPartition(y = iphoneNZV$iphonesentiment, 
                                  ## ## the outcome data are needed
                                  p =.70, 
                                  ## ## The percentage of data in the
                                  ##  ## training set
                                  list = FALSE)
## The format of the results
## The output is a set of integers that belong in the training set.

# to partition the data into training and testing datasets
trainingNZV <- iphoneNZV[ inTrainNZV,]
testingNZV <- iphoneNZV[-inTrainNZV,]
nrow(trainingNZV)
# verify number of obs 
str(trainingNZV)  
str(testingNZV) 

#--- Dataset RFE features removed iphone---#

inTrainRFE <- createDataPartition(y = iphoneRFE$iphonesentiment, 
                                  ## ## the outcome data are needed
                                  p =.70, 
                                  ## ## The percentage of data in the
                                  ##  ## training set
                                  list = FALSE)
## The format of the results
## The output is a set of integers that belong in the training set.

# to partition the data into training and testing datasets
trainingRFE <- iphoneRFE[ inTrainRFE,]
testingRFE <- iphoneRFE[-inTrainRFE,]
nrow(trainingRFE)
nrow(testingRFE)
# verify number of obs 
str(trainingRFE)  
str(testingRFE) 

#--- Dataset iphone target reingeneered ---#

inTrainRC <- createDataPartition(y = iphoneRC$iphonesentiment, 
                                  ## ## the outcome data are needed
                                  p =.70, 
                                  ## ## The percentage of data in the
                                  ##  ## training set
                                  list = FALSE)
## The format of the results
## The output is a set of integers that belong in the training set.

# to partition the data into training and testing datasets
trainingRC <- iphoneRC[ inTrainRC,]
testingRC <- iphoneRC[-inTrainRC,]
nrow(trainingRC)
nrow(testingRC)
# verify number of obs 
str(trainingRC)  
str(testingRC) 

#--- Dataset samsung galaxy , no features removed ---#
inTrainALLsam <- createDataPartition(y = samsungDF$galaxysentiment, 
                                  ## ## the outcome data are needed
                                  p =.70, 
                                  ## ## The percentage of data in the
                                  ##  ## training set
                                  list = FALSE)
## The format of the results
## The output is a set of integers that belong in the training set.

# to partition the data into training and testing datasets
trainingALLsam <- samsungDF[ inTrainALLsam,]
testingALLsam <- samsungDF[-inTrainALLsam,]
nrow(trainingALLsam)
# verify number of obs 
str(trainingALLsam)  
str(testingALLsam) 

#--- Dataset correleted features removed Samsung ---#

inTrainCORsam <- createDataPartition(y = samsungCOR$galaxysentiment, 
                                  ## ## the outcome data are needed
                                  p =.70, 
                                  ## ## The percentage of data in the
                                  ##  ## training set
                                  list = FALSE)
## The format of the results
## The output is a set of integers that belong in the training set.

# to partition the data into training and testing datasets
trainingCORsam <- samsungCOR[ inTrainCORsam,]
testingCORsam <- samsungCOR[-inTrainCORsam,]
nrow(trainingCORsam)
nrow(testingCORsam)
# verify number of obs 
str(trainingCORsam)  
str(testingCORsam)


#--- Dataset near zero variance features removed Samsung galaxy ---#

inTrainNZVsam <- createDataPartition(y = samsungNZV$galaxysentiment, 
                                  ## ## the outcome data are needed
                                  p =.70, 
                                  ## ## The percentage of data in the
                                  ##  ## training set
                                  list = FALSE)
## The format of the results
## The output is a set of integers that belong in the training set.

# to partition the data into training and testing datasets
trainingNZVsam <- samsungNZV[ inTrainNZVsam,]
testingNZVsam <- samsungNZV[-inTrainNZVsam,]
nrow(trainingNZVsam)
# verify number of obs 
str(trainingNZVsam)  
str(testingNZVsam) 

#--- Dataset RFE features removed samsung---#

inTrainRFEsam <- createDataPartition(y = samsungRFE$galaxysentiment, 
                                  ## ## the outcome data are needed
                                  p =.70, 
                                  ## ## The percentage of data in the
                                  ##  ## training set
                                  list = FALSE)
## The format of the results
## The output is a set of integers that belong in the training set.

# to partition the data into training and testing datasets
trainingRFEsam <- samsungRFE[ inTrainRFEsam,]
testingRFEsam <- samsungRFE[-inTrainRFEsam,]
nrow(trainingRFEsam)
nrow(testingRFEsam)
# verify number of obs 
str(trainingRFEsam)  
str(testingRFEsam) 

#--- Dataset iphone target reingeneered ---#

inTrainRCSam <- createDataPartition(y = samsungRC$galaxysentiment, 
                                 ## ## the outcome data are needed
                                 p =.70, 
                                 ## ## The percentage of data in the
                                 ##  ## training set
                                 list = FALSE)
## The format of the results
## The output is a set of integers that belong in the training set.

# to partition the data into training and testing datasets
trainingRCSam <- samsungRC[ inTrainRCSam,]
testingRCSam <- samsungRC[-inTrainRCSam,]
nrow(trainingRCSam)
nrow(testingRCSam)
# verify number of obs 
str(trainingRCSam)  
str(testingRCSam) 

#--- Dataset samsung target reingeneered ---#

inTrainRCSam <- createDataPartition(y = samsungRC$galaxysentiment, 
                                    ## ## the outcome data are needed
                                    p =.70, 
                                    ## ## The percentage of data in the
                                    ##  ## training set
                                    list = FALSE)
## The format of the results
## The output is a set of integers that belong in the training set.

# to partition the data into training and testing datasets
trainingRCSam <- samsungRC[ inTrainRCSam,]
testingRCSam <- samsungRC[-inTrainRCSam,]
nrow(trainingRCSam)
nrow(testingRCSam)
# verify number of obs 
str(trainingRCSam)  
str(testingRCSam) 
################
# Train control
################

# trainControl modifies the resampling method.  Default maethod is boot method
# repeatedcv specifies the repeated K-fold cross-validation and repeats
# controls the # of repetitions
# set 10 fold cross validation for repeatedcsv

fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)


########################
# Train model for iphone
########################

## ------- C5.0 ------- ##

modelLookup('C5.0')

# train algorithm is used to tune a model
# tuneLength controls how many parameter values are evaluated
# use tuneGrid instead when specific values are needed

#--- iphoneDF, no features removed ---#

c5_all <-  train(iphonesentiment ~ .,
                data = trainingALL,
                method = "C5.0",
                tuneLength = 2,
                trControl = fitControl,
                # na.action = na.pass)
                na.action = na.omit)

c5_all

# C5.0 

# 9083 samples
# 58 predictor
# 6 classes: 'very_negative', 'negative', 'somewhat_negative', 'somewhat_positive', 'positive', 'very_positive' 

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8174, 8173, 8174, 8175, 8175, 8175, ... 
# Resampling results across tuning parameters:
#   model  winnow  trials  Accuracy   Kappa    
# rules  FALSE    1      0.7712433  0.5549480
# rules  FALSE   10      0.7657051  0.5467114
# rules   TRUE    1      0.7705167  0.5536801
# rules   TRUE   10      0.7641418  0.5441686
# tree   FALSE    1      0.7701312  0.5537443
# tree   FALSE   10      0.7642516  0.5450872
# tree    TRUE    1      0.7699773  0.5534878
# tree    TRUE   10      0.7632057  0.5433720
  
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = rules and winnow = FALSE.
  
c5.0Classes <- predict(c5_all, newdata = testingALL)
str(c5.0Classes)
# Factor w/ 6 levels "very_negative",..: 5 6 6 6 6 6 6 6 6 6 ...


postResample(c5.0Classes,testingALL$iphonesentiment)
# Accuracy     Kappa 
# 0.7730077 0.5590862 

# Using the option type = "prob" can be used to compute class
# probabilities from the model. 

c5.0Probs <- predict(c5_all, newdata = testingALL, type = "prob")

head(c5.0Probs)

# very_negative negative somewhat_negative somewhat_positive  positive very_positive
# 7              0        0                 0                 0 0.7461476     0.2538524
# 8              0        0                 0                 0 0.0000000     1.0000000
# 17             0        0                 0                 0 0.0000000     1.0000000
# 30             0        0                 0                 0 0.0000000     1.0000000
# 33             0        0                 0                 0 0.0000000     1.0000000
# 34             0        0                 0                 0 0.0000000     1.0000000


plot(c5_all)

# confusion matrix and associated statistics for the model fit
## !!!!!!! used for classification not regression 

confusionMatrix(data = c5.0Classes, testingALL$iphonesentiment)
# Confusion Matrix and Statistics

# Reference
# Prediction          very_negative negative somewhat_negative somewhat_positive positive very_positive
# very_negative               378        0                 1                 4        6             8
# negative                      0        0                 0                 0        0             0
# somewhat_negative             2        0                18                 0        0             0
# somewhat_positive             5        0                 1               226        3             4
# positive                      1        0                 2                 1      146            11
# very_positive               202      117               114               125      276          2239

# Overall Statistics

# Accuracy : 0.773           
# 95% CI : (0.7595, 0.7861)
# No Information Rate : 0.5815          
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.5591          
# Mcnemar's Test P-Value : NA   

# Statistics by Class:

#                          Class: very_negative Class: negative Class: somewhat_negative Class: somewhat_positive
# Sensitivity                       0.64286         0.00000                 0.132353                  0.63483
# Specificity                       0.99425         1.00000                 0.999467                  0.99632
# Pos Pred Value                    0.95214             NaN                 0.900000                  0.94561
# Neg Pred Value                    0.93988         0.96992                 0.969509                  0.96439
# Prevalence                        0.15116         0.03008                 0.034961                  0.09152
# Detection Rate                    0.09717         0.00000                 0.004627                  0.05810
# Detection Prevalence              0.10206         0.00000                 0.005141                  0.06144
# Balanced Accuracy                 0.81855         0.50000                 0.565910                  0.81558
#                          Class: positive Class: very_positive
# Sensitivity                  0.33875               0.9898
# Specificity                  0.99566               0.4877
# Pos Pred Value               0.90683               0.7286
# Neg Pred Value               0.92357               0.9718
# Prevalence                   0.11080               0.5815
# Detection Rate               0.03753               0.5756
# Detection Prevalence         0.04139               0.7900

# var imp
c5.0imp = varImp(c5_all, scale = FALSE)

# summarize importance
print(c5.0imp)
# C5.0 variable importance

# only 20 most important variables shown (out of 58)
# Overall
# iphone         100.00
# iphoneperpos    10.13
# googleandroid    9.25
# iphonecampos     6.97
# iphonedisneg     6.78
# iphoneperunc     5.15
# htccamunc        5.13
# sonyxperia       4.77
# iphonedisunc     4.49
# iphonedispos     4.39
# iphonecamneg     3.92
# iphonecamunc     3.26
# samsunggalaxy    3.26
# samsungperneg    2.58
# htcphone         2.21
# iphoneperneg     0.15
# ios              0.00
# samsungdisunc    0.00
# nokiacamneg      0.00
# googleperneg     0.00

# plot importance
plot(c5.0imp)

#--- iphoneCORR, correlated features removed ---#

c5_cor <-  train(iphonesentiment ~ .,
                 data = trainingCOR,
                 method = "C5.0",
                 tuneLength = 2,
                 trControl = fitControl,
                 # na.action = na.pass)
                 na.action = na.omit)

c5_cor
# C5.0 

#9083 samples
#45 predictor
# 6 classes: 'very_negative', 'negative', 'somewhat_negative', 'somewhat_positive', 'positive', 'very_positive' 

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8175, 8175, 8175, 8174, 8175, 8175, ... 
# Resampling results across tuning parameters:
#   model  winnow  trials  Accuracy   Kappa    
# rules  FALSE    1      0.7684332  0.5495705
# rules  FALSE   10      0.7636770  0.5431320
# rules   TRUE    1      0.7682467  0.5492911
# rules   TRUE   10      0.7639638  0.5436002
# tree   FALSE    1      0.7680258  0.5497947
# tree   FALSE   10      0.7631924  0.5431925
# tree    TRUE    1      0.7674427  0.5485567
# tree    TRUE   10      0.7627307  0.5424001

# Accuracy was used to select the optimal model using the
# largest value.
# The final values used for the model were trials = 1, model =
#   rules and winnow = FALSE.

c5.0ClassesC <- predict(c5_cor, newdata = testingCOR)
str(c5.0ClassesC)
# Factor w/ 6 levels "very_negative",..: 5 6 4 6 6 6 6 6 6 6 ...

postResample(c5.0ClassesC,testingCOR$iphonesentiment)
# Accuracy     Kappa 
# 0.7766067 0.5669767

# Using the option type = "prob" can be used to compute class
# probabilities from the model. 

c5.0ProbsC <- predict(c5_cor, newdata = testingCOR, type = "prob")

head(c5.0ProbsC)

#        very_negative negative somewhat_negative somewhat_positive  positive very_positive
# 7              0        0                 0         0.0000000 0.8160141     0.1839859
# 8              0        0                 0         0.0000000 0.0000000     1.0000000
# 11             0        0                 0         0.7538401 0.0000000     0.2461599
# 17             0        0                 0         0.0000000 0.0000000     1.0000000
# 25             0        0                 0         0.0000000 0.0000000     1.0000000
# 27             0        0                 0         0.0000000 0.0000000     1.0000000

plot(c5_cor)

# confusion matrix and associated statistics for the model fit
## !!!!!!! used for classification not regression 

confusionMatrix(data = c5.0ClassesC, testingCOR$iphonesentiment)

# Confusion Matrix and Statistics

# Reference
# Prediction          very_negative negative somewhat_negative somewhat_positive positive very_positive
# very_negative               385        0                 1                 3        5             5
# negative                      0        0                 0                 0        0             0
# somewhat_negative             1        0                23                 0        0             0
# somewhat_positive             3        1                 0               233        3             8
# positive                      1        2                 1                 1      140             9
# very_positive               198      114               111               119      283          2240

# Overall Statistics

# Accuracy : 0.7766          
# 95% CI : (0.7632, 0.7896)
# No Information Rate : 0.5815          
# P-Value [Acc > NIR] : < 2.2e-16   
# Kappa : 0.567           
# Mcnemar's Test P-Value : NA              

# Statistics by Class:

# Class: very_negative Class: negative Class: somewhat_negative Class: somewhat_positive
# Sensitivity                       0.65476         0.00000                 0.169118                  0.65449
# Specificity                       0.99576         1.00000                 0.999734                  0.99576
# Pos Pred Value                    0.96491             NaN                 0.958333                  0.93952
# Neg Pred Value                    0.94185         0.96992                 0.970771                  0.96623
# Prevalence                        0.15116         0.03008                 0.034961                  0.09152
# Detection Rate                    0.09897         0.00000                 0.005913                  0.05990
# Detection Prevalence              0.10257         0.00000                 0.006170                  0.06375
# Balanced Accuracy                 0.82526         0.50000                 0.584426                  0.82512
# Class: positive Class: very_positive
# Sensitivity                  0.32483               0.9903
# Specificity                  0.99595               0.4932
# Pos Pred Value               0.90909               0.7308
# Neg Pred Value               0.92211               0.9733
# Prevalence                   0.11080               0.5815
# Detection Rate               0.03599               0.5758
# Detection Prevalence         0.03959               0.7879
# Balanced Accuracy            0.66039               0.7418

# var imp
c5.0impC = varImp(c5_cor, scale = FALSE)


# summarize importance
print(c5.0impC)

# C5.0 variable importance

# only 20 most important variables shown (out of 45)

# Overall
# iphone         100.00
# iphoneperpos    10.40
# iphonedisneg     9.89
# googleandroid    8.73
# iphonedispos     8.18
# htccampos        6.95
# iphonedisunc     5.01
# iphonecampos     4.58
# sonyxperia       4.17
# samsunggalaxy    3.41
# iphonecamneg     3.34
# samsungperneg    3.28
# iphoneperunc     2.96
# iphoneperneg     2.35
# iphonecamunc     1.94
# htcdisneg        0.00
# samsungperpos    0.00
# htcperpos        0.00
# sonycamneg       0.00
# nokialumina      0.00

# plot importance


plot(c5.0impC)

#--- iphoneNZV nearly zero variance features removed ---#

c5_nze <-  train(iphonesentiment ~ .,
                 data = trainingNZV,
                 method = "C5.0",
                 tuneLength = 2,
                 trControl = fitControl,
                 # na.action = na.pass)
                 na.action = na.omit)

c5_nze

# C5.0 

# 9083 samples
# 11 predictor
# 6 classes: 'very_negative', 'negative', 'somewhat_negative', 'somewhat_positive', 'positive', 'very_positive' 

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8175, 8176, 8173, 8173, 8176, 8174, ... 
# Resampling results across tuning parameters:
  
#   model  winnow  trials  Accuracy   Kappa    
# rules  FALSE    1      0.7542126  0.5163908
# rules  FALSE   10      0.7441505  0.4959812
# rules   TRUE    1      0.7536404  0.5155380
# rules   TRUE   10      0.7433027  0.4947381
# tree   FALSE    1      0.7546858  0.5177828
# tree   FALSE   10      0.7448211  0.4979571
# tree    TRUE    1      0.7542786  0.5173111
# tree    TRUE   10      0.7435224  0.4955068

# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = tree and winnow = FALSE.

c5.0Classesnzv <- predict(c5_nze, newdata = testingNZV)
str(c5.0Classesnzv)

# Factor w/ 6 levels "very_negative",..: 6 6 6 5 6 6 1 6 6 6 ...

postResample(c5.0Classesnzv,testingNZV$iphonesentiment)
# Accuracy     Kappa 
# 0.7645244 0.5408104 

# Using the option type = "prob" can be used to compute class
# probabilities from the model. 

c5.0Probsnze <- predict(c5_nze, newdata = testingNZV, type = "prob")

head(c5.0Probsnze)
# very_negative     negative somewhat_negative somewhat_positive   positive very_positive
# 3    0.066042543 0.0374651553      0.0367940543      0.0566254989 0.08854543    0.71452731
# 4    0.066042543 0.0374651553      0.0367940543      0.0566254989 0.08854543    0.71452731
# 5    0.066042543 0.0374651553      0.0367940543      0.0566254989 0.08854543    0.71452731
# 7    0.009210173 0.0002404492      0.0002800837      0.0007327976 0.97688781    0.01264868
# 9    0.066042543 0.0374651553      0.0367940543      0.0566254989 0.08854543    0.71452731
# 10   0.066042543 0.0374651553      0.0367940543      0.0566254989 0.08854543    0.71452731


plot(c5_nze)

# confusion matrix and associated statistics for the model fit
## !!!!!!! used for classification not regression 

confusionMatrix(data = c5.0Classesnzv, testingNZV$iphonesentiment)

# Confusion Matrix and Statistics

# Reference
# Prediction          very_negative negative somewhat_negative somewhat_positive positive very_positive
# very_negative               402        0                14                 7        2             7
# negative                      0        0                 0                 0        0             0
# somewhat_negative             0        0                 0                 0        0             0
# somewhat_positive             4        0                 2               188        5             5
# positive                      5        0                 3                 0      149            15
# very_positive               177      117               117               161      275          2235

# Overall Statistics

# Accuracy : 0.7645          
# 95% CI : (0.7509, 0.7778)
# No Information Rate : 0.5815          
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.5408          
# Mcnemar's Test P-Value : NA              

# Statistics by Class:
# Class: very_negative Class: negative Class: somewhat_negative Class: somewhat_positive
# Sensitivity                        0.6837         0.00000                  0.00000                  0.52809
# Specificity                        0.9909         1.00000                  1.00000                  0.99547
# Pos Pred Value                     0.9306             NaN                      NaN                  0.92157
# Neg Pred Value                     0.9462         0.96992                  0.96504                  0.95442
# Prevalence                         0.1512         0.03008                  0.03496                  0.09152
# Detection Rate                     0.1033         0.00000                  0.00000                  0.04833
# Detection Prevalence               0.1111         0.00000                  0.00000                  0.05244
# Balanced Accuracy                  0.8373         0.50000                  0.50000                  0.76178
# Class: positive Class: very_positive
# Sensitivity                  0.34571               0.9881
# Specificity                  0.99335               0.4797
# Pos Pred Value               0.86628               0.7252
# Neg Pred Value               0.92415               0.9666
# Prevalence                   0.11080               0.5815
# Detection Rate               0.03830               0.5746
# Detection Prevalence         0.04422               0.7923
# Balanced Accuracy            0.66953               0.7339

# var imp
c5.0impnzv = varImp(c5_nze, scale = FALSE)

# summarize importance
print(c5.0impnzv)
# C5.0 variable importance

# Overall
# iphone         100.00
# samsunggalaxy   92.62
# htcphone        92.62
# iphonedisunc    87.94
# iphonedispos    76.82
# iphoneperpos    24.68
# iphonedisneg    11.95
# iphonecampos    11.75
# iphonecamunc    10.34
# iphoneperunc     3.75
# iphoneperneg     2.46

# plot importance
plot(c5.0impnzv)

#--- iphoneRFE  features removed ---#

c5_rfe <-  train(iphonesentiment ~ .,
                 data = trainingRFE,
                 method = "C5.0",
                 tuneLength = 2,
                 trControl = fitControl,
                 # na.action = na.pass)
                 na.action = na.omit)

c5_rfe

# C5.0 

# 9083 samples
# 21 predictor
# 6 classes: 'very_negative', 'negative', 'somewhat_negative', 'somewhat_positive', 'positive', 'very_positive' 

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8175, 8173, 8174, 8176, 8176, 8175, ... 
# Resampling results across tuning parameters:
  
#   model  winnow  trials  Accuracy   Kappa    
# rules  FALSE    1      0.7681730  0.5479132
# rules  FALSE   10      0.7626343  0.5398629
# rules   TRUE    1      0.7680522  0.5477238
# rules   TRUE   10      0.7617429  0.5379065
# tree   FALSE    1      0.7666427  0.5460251
# tree   FALSE   10      0.7612474  0.5382279
# tree    TRUE    1      0.7667309  0.5461323
# tree    TRUE   10      0.7611381  0.5376662

# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = rules and winnow = FALSE.

c5.0Classesrfe <- predict(c5_rfe, newdata = testingRFE)
str(c5.0Classesrfe)
# Factor w/ 6 levels "very_negative",..: 6 6 6 6 6 6 6 6 6 1 ...

postResample(c5.0Classesrfe,testingRFE$iphonesentiment)
# Accuracy     Kappa 
# 0.7809769 0.5767989 

# Using the option type = "prob" can be used to compute class
# probabilities from the model. 

c5.0Probsrfe <- predict(c5_rfe, newdata = testingRFE, type = "prob")

head(c5.0Probsrfe)

# very_negative negative somewhat_negative somewhat_positive positive very_positive
# 1              0        0                 0        0.06639839        0     0.9336016
# 5              0        0                 0        0.04280156        0     0.9571984
# 9              0        0                 0        0.06639839        0     0.9336016
# 13             0        0                 0        0.06639839        0     0.9336016
# 14             0        0                 0        0.06639839        0     0.9336016
# 15             0        0                 0        0.06639839        0     0.9336016

plot(c5_rfe)

# confusion matrix and associated statistics for the model fit
## !!!!!!! used for classification not regression 

confusionMatrix(data = c5.0Classesrfe, testingRFE$iphonesentiment)

# Confusion Matrix and Statistics

# Reference
# Prediction          very_negative negative somewhat_negative somewhat_positive positive very_positive
# very_negative               379        0                 1                 2        3             7
# negative                      0        0                 0                 0        0             0
# somewhat_negative             0        0                20                 0        0             0
# somewhat_positive            11        0                 1               244        3             5
# positive                      3        1                 1                 2      150             5
# very_positive               195      116               113               108      275          2245

# Overall Statistics

# Accuracy : 0.781           
# 95% CI : (0.7676, 0.7939)
# No Information Rate : 0.5815          
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.5768          
# Mcnemar's Test P-Value : NA              

# Statistics by Class:
  
#   Class: very_negative Class: negative Class: somewhat_negative Class: somewhat_positive
# Sensitivity                       0.64456         0.00000                 0.147059                  0.68539
# Specificity                       0.99606         1.00000                 1.000000                  0.99434
# Pos Pred Value                    0.96684             NaN                 1.000000                  0.92424
# Neg Pred Value                    0.94025         0.96992                 0.970026                  0.96911
# Prevalence                        0.15116         0.03008                 0.034961                  0.09152
# Detection Rate                    0.09743         0.00000                 0.005141                  0.06272
# Detection Prevalence              0.10077         0.00000                 0.005141                  0.06787
# Balanced Accuracy                 0.82031         0.50000                 0.573529                  0.83987
# Class: positive Class: very_positive
# Sensitivity                  0.34803               0.9925
# Specificity                  0.99653               0.5043
# Pos Pred Value               0.92593               0.7356
# Neg Pred Value               0.92462               0.9797
# Prevalence                   0.11080               0.5815
# Detection Rate               0.03856               0.5771
# Detection Prevalence         0.04165               0.7846
# Balanced Accuracy            0.67228               0.7484

# var imp
c5.0imprfe = varImp(c5_rfe, scale = FALSE)

c5.0imprfe

# C5.0 variable importance

# only 20 most important variables shown (out of 21)

# Overall
# iphone         100.00
# googleandroid   90.39
# htccampos       86.54
# sonyxperia      84.95
# iphonedisneg    84.10
# ios             83.58
# samsunggalaxy   81.86
# iphoneperpos    20.30
# iphonedisunc    17.36
# samsungperpos   13.27
# iphoneperneg    13.07
# samsungperunc    7.06
# iphonecamneg     6.65
# iphonecampos     6.24
# iphonedispos     4.79
# htcphone         4.36
# iphoneperunc     3.21
# iphonecamunc     0.24
# htcdisneg        0.00
# htccamneg        0.00


# plot importance
plot(c5.0imprfe)

#--- iphoneDF, no features removed recoded sentiment target---#
c5_rc <-  train(iphonesentiment ~ .,
                 data = trainingRC,
                 method = "C5.0",
                 tuneLength = 2,
                 trControl = fitControl,
                 # na.action = na.pass)
                 na.action = na.omit)

c5_rc

# C5.0 

# 9083 samples
# 58 predictor
# 4 classes: 'negative', 'somewhat_negative', 'somewhat_positive', 'positive' 

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8175, 8176, 8174, 8174, 8174, 8174, ... 
# Resampling results across tuning parameters:
  
#   model  winnow  trials  Accuracy   Kappa    
# rules  FALSE    1      0.8488055  0.6228818
# rules  FALSE   10      0.8468457  0.6210302
# rules   TRUE    1      0.8489595  0.6233491
# rules   TRUE   10      0.8456012  0.6172571
# tree   FALSE    1      0.8486732  0.6236460
# tree   FALSE   10      0.8454809  0.6190772
# tree    TRUE    1      0.8492897  0.6250556
# tree    TRUE   10      0.8450290  0.6174536

# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = tree and winnow = TRUE.

c5.0Classesrc <- predict(c5_rc, newdata = testingRC)
str(c5.0Classesrc)
# Factor w/ 4 levels "negative","somewhat_negative",..: 4 4 4 1 4 4 4 4 4 4 ...

postResample(c5.0Classesrc,testingRC$iphonesentiment)
# Accuracy     Kappa 
# 0.8401028 0.5978886 

# Using the option type = "prob" can be used to compute class
# probabilities from the model. 

c5.0Probsrc <- predict(c5_rc, newdata = testingRC, type = "prob")

head(c5.0Probsrc)
# negative somewhat_negative somewhat_positive   positive
# 1  0.10329801      0.0375884570      0.0404759772 0.81863756
# 3  0.10329801      0.0375884570      0.0404759772 0.81863756
# 4  0.10329801      0.0375884570      0.0404759772 0.81863756
# 12 0.98105550      0.0003646923      0.0009541635 0.01762565
# 16 0.10329801      0.0375884570      0.0404759772 0.81863756
# 19 0.08678885      0.0322608349      0.0242614286 0.85668889

plot(c5_rc)

# confusion matrix and associated statistics for the model fit
## !!!!!!! used for classification not regression 

confusionMatrix(data = c5.0Classesrc, testingRC$iphonesentiment)
# Confusion Matrix and Statistics

# Reference
# Prediction          negative somewhat_negative somewhat_positive positive
# negative               364                 0                 9       10
# somewhat_negative        0                16                 0        0
# somewhat_positive        3                 3               219       14
# positive               338               117               128     2669

# Overall Statistics

# Accuracy : 0.8401          
# 95% CI : (0.8282, 0.8515)
# No Information Rate : 0.6923          
# P-Value [Acc > NIR] : < 2.2e-16 

# Kappa : 0.5979          
# Mcnemar's Test P-Value : NA              

# Statistics by Class:

# Class: negative Class: somewhat_negative Class: somewhat_positive Class: positive
# Sensitivity                  0.51631                 0.117647                  0.61517          0.9911
# Specificity                  0.99403                 1.000000                  0.99434          0.5129
# Pos Pred Value               0.95039                 1.000000                  0.91632          0.8207
# Neg Pred Value               0.90277                 0.969024                  0.96248          0.9624
# Prevalence                   0.18123                 0.034961                  0.09152          0.6923
# Detection Rate               0.09357                 0.004113                  0.05630          0.6861
# Detection Prevalence         0.09846                 0.004113                  0.06144          0.8360
# Balanced Accuracy            0.75517                 0.558824                  0.80475          0.7520

# var imp
c5.0imprc = varImp(c5_rc, scale = FALSE)

# plot importance
c5.0imprc

# C5.0 variable importance

# only 20 most important variables shown (out of 58)

# Overall
# iphone         100.00
# googleandroid   98.90
# samsungcamneg   92.47
# sonyxperia      89.90
# iphonedispos    88.09
# samsunggalaxy   87.94
# htccampos       86.13
# iphonedisunc    71.56
# iphoneperpos    24.78
# iphonedisneg     8.94
# iphonecampos     2.47
# iphoneperunc     2.30
# iphonecamunc     1.84
# ios              0.53
# htcperpos        0.00
# sonyperpos       0.00
# googleperunc     0.00
# nokialumina      0.00
# samsungcampos    0.00
# sonycamunc       0.00
# iphonedisneg     8.94

plot(c5.0imprc)



#--- iphone, Principal Component Analysis PCA ---#
c5_pca <-  train(iphonesentiment ~ .,
                data = train.pca,
                method = "C5.0",
                tuneLength = 2,
                trControl = fitControl,
                # na.action = na.pass)
                na.action = na.omit)

c5_pca

# C5.0 

# 9083 samples
# 25 predictor
# 6 classes: 'very_negative', 'negative', 'somewhat_negative', 'somewhat_positive', 'positive', 'very_positive' 

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8177, 8174, 8175, 8174, 8176, 8174, ...
# Resampling results across tuning parameters:
  
#   model  winnow  trials  Accuracy   Kappa    
# rules  FALSE    1      0.7580446  0.5299395
# rules  FALSE   10      0.7565692  0.5297176
# rules   TRUE    1      0.7592330  0.5320031
# rules   TRUE   10      0.7562612  0.5289358
# tree   FALSE    1      0.7581101  0.5312632
# tree   FALSE   10      0.7548627  0.5260858
# tree    TRUE    1      0.7591340  0.5329973
# tree    TRUE   10      0.7551158  0.5264384

# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = rules and winnow = TRUE.

c5.0Classespca <- predict(c5_pca, newdata = test.pca)
str(c5.0Classespca)

# Factor w/ 6 levels "very_negative",..: 5 6 6 6 6 6 6 6 6 6 ...

postResample(c5.0Classespca,test.pca$iphonesentiment)
# Accuracy     Kappa 
# 0.7622108 0.5397932 

# Using the option type = "prob" can be used to compute class
# probabilities from the model. 

c5.0Probspca <- predict(c5_pca, newdata = test.pca, type = "prob")

head(c5.0Probspca)

# very_negative negative somewhat_negative somewhat_positive  positive very_positive
# 7              0        0                 0                 0 0.7483521     0.2516479
# 8              0        0                 0                 0 0.0000000     1.0000000
# 17             0        0                 0                 0 0.0000000     1.0000000
# 30             0        0                 0                 0 0.0000000     1.0000000
# 33             0        0                 0                 0 0.0000000     1.0000000
# 34             0        0                 0                 0 0.0000000     1.0000000

plot(c5_pca)
# confusion matrix and associated statistics for the model fit
## !!!!!!! used for classification not regression 

confusionMatrix(data = c5.0Classespca, test.pca$iphonesentiment)

# Confusion Matrix and Statistics

# Reference
# Prediction          very_negative negative somewhat_negative somewhat_positive positive very_positive
# very_negative               387        0                 3                15       16            27
# negative                      0        0                 0                 0        0             0
# somewhat_negative             1        0                18                 0        0             0
# somewhat_positive             1        0                 2               206        0             7
# positive                      5        2                 0                 0      137            11
# very_positive               194      115               113               135      278          2217

# Overall Statistics

# Accuracy : 0.7622          
# 95% CI : (0.7485, 0.7755)
# No Information Rate : 0.5815          
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.5398          
# Mcnemar's Test P-Value : NA  
# Statistics by Class:

# Class: very_negative Class: negative Class: somewhat_negative Class: somewhat_positive
# Sensitivity                       0.65816         0.00000                 0.132353                  0.57865
# Specificity                       0.98153         1.00000                 0.999734                  0.99717
# Pos Pred Value                    0.86384             NaN                 0.947368                  0.95370
# Neg Pred Value                    0.94160         0.96992                 0.969517                  0.95917
# Prevalence                        0.15116         0.03008                 0.034961                  0.09152
# Detection Rate                    0.09949         0.00000                 0.004627                  0.05296
# Detection Prevalence              0.11517         0.00000                 0.004884                  0.05553
# Balanced Accuracy                 0.81984         0.50000                 0.566043                  0.78791
# Class: positive Class: very_positive
# Sensitivity                  0.31787               0.9801
# Specificity                  0.99480               0.4871
# Pos Pred Value               0.88387               0.7264
# Neg Pred Value               0.92129               0.9463
# Prevalence                   0.11080               0.5815
# Detection Rate               0.03522               0.5699
# Detection Prevalence         0.03985               0.7846
# Balanced Accuracy            0.65633               0.7336

# var imp
c5.0imppca = varImp(c5_pca, scale = FALSE)

# plot importance
c5.0imppca

# C5.0 variable importance

# only 20 most important variables shown (out of 25)

# Overall
# PC5   100.00
# PC4     7.42
# PC10    6.87
# PC7     5.78
# PC23    5.21
# PC3     3.81
# PC16    3.28
# PC12    3.12
# PC24    3.08
# PC22    2.90
# PC14    2.61
# PC15    1.55
# PC25    0.70
# PC8     0.51
# PC1     0.50
# PC11    0.43
# PC13    0.35
# PC2     0.34
# PC17    0.26
# PC6     0.10

plot(c5.0imppca)

#--- Dataset 1 iphone re-engineer Principal Component Analysis PCA and recoded target ---#
c5_pcarc <-  train(iphonesentiment ~ .,
                 data = train.pcaRC,
                 method = "C5.0",
                 tuneLength = 2,
                 trControl = fitControl,
                 # na.action = na.pass)
                 na.action = na.omit)

c5_pcarc

# C5.0 

# 9083 samples
# 27 predictor
# 4 classes: 'negative', 'somewhat_negative', 'somewhat_positive', 'positive' 

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8175, 8176, 8174, 8176, 8175, 8174, ... 
# Resampling results across tuning parameters:
  
#   model  winnow  trials  Accuracy   Kappa    
# rules  FALSE    1      0.8407795  0.6014869
# rules  FALSE   10      0.8408347  0.6036230
# rules   TRUE    1      0.8413747  0.6030573
# rules   TRUE   10      0.8404272  0.6028401
# tree   FALSE    1      0.8404163  0.6008139
# tree   FALSE   10      0.8405927  0.6031072
# tree    TRUE    1      0.8408898  0.6023701
# tree    TRUE   10      0.8405483  0.6033258

# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = rules and winnow = TRUE.

c5.0Classespcarc <- predict(c5_pcarc, newdata = test.pcaRC)
str(c5.0Classespcarc)

# Factor w/ 4 levels "negative","somewhat_negative",..: 4 4 4 1 4 4 4 4 4 4 ...

postResample(c5.0Classespcarc,test.pcaRC$iphonesentiment)
# Accuracy     Kappa 
# 0.8354756 0.5854161 

# Using the option type = "prob" can be used to compute class
# probabilities from the model. 

c5.0Probspcarc <- predict(c5_pcarc, newdata = test.pcaRC, type = "prob")

head(c5.0Probspcarc)

# negative somewhat_positive positive
# 1         0                 0                 0        1
# 3         0                 0                 0        1
# 4         0                 0                 0        1
# 12        1                 0                 0        0
# 16        0                 0                 0        1
# 19        0                 0                 0        1
plot(c5_pcarc)
# confusion matrix and associated statistics for the model fit
## !!!!!!! used for classification not regression 

confusionMatrix(data = c5.0Classespcarc, test.pcaRC$iphonesentiment)
# confusion Matrix and Statistics

# Reference
# Prediction          negative somewhat_negative somewhat_positive positive
# negative               368                 2                10       25
# somewhat_negative        0                15                 0        2
# somewhat_positive        5                 0               205        4
# positive               332               119               141     2662

# Overall Statistics

# Accuracy : 0.8355         
# 95% CI : (0.8234, 0.847)
# No Information Rate : 0.6923         
# P-Value [Acc > NIR] : < 2.2e-16      

# Kappa : 0.5854         
# Mcnemar's Test P-Value : NA    


# Statistics by Class:

# Class: negative Class: somewhat_negative Class: somewhat_positive Class: positive
# Sensitivity                   0.5220                 0.110294                  0.57584          0.9885
# Specificity                   0.9884                 0.999467                  0.99745          0.5054
# Pos Pred Value                0.9086                 0.882353                  0.95794          0.8181
# Neg Pred Value                0.9033                 0.968758                  0.95892          0.9513
# Prevalence                    0.1812                 0.034961                  0.09152          0.6923
# Detection Rate                0.0946                 0.003856                  0.05270          0.6843
# Detection Prevalence          0.1041                 0.004370                  0.05501          0.8365
# Balanced Accuracy             0.7552                 0.554881                  0.78665          0.7470
 

# var imp
c5.0imppcarc = varImp(c5_pcarc, scale = FALSE)

# plot importance
c5.0imppca

plot(c5.0imppca)
## ------- RF Random Forest ------- ##
modelLookup('rf')
#--- iphoneDF, no features removed ---#

rfAll<-train(iphonesentiment~.,data=trainingALL,method="rf",
                trControl= fitControl,
                prox=TRUE,allowParallel=TRUE)

rfAll
# Random Forest 

# 9083 samples
# 58 predictor
# 6 classes: 'very_negative', 'negative', 'somewhat_negative', 'somewhat_positive', 'positive', 'very_positive' 

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8175, 8173, 8175, 8176, 8175, 8175, ... 
# Resampling results across tuning parameters:
  
#   mtry  Accuracy   Kappa    
# 2    0.6982390  0.3650603
# 30    0.7709129  0.5581979
# 58    0.7621061  0.5453599

# Accuracy was used to select the optimal model using the
# largest value.
# The final value used for the model was mtry = 30.


# to predict new samples use predict.train
rfClasses <- predict(rfAll, newdata = testingALL)
str(rfClasses)
# Factor w/ 6 levels "very_negative",..: 5 6 6 6 6 6 6 6 6 6 ...

postResample(rfClasses,testingALL$iphonesentiment)
# Accuracy     Kappa 
# 0.7768638 0.5703315

# plot(rf_Fit) takes the output of the train object and creates a plot
plot(rfAll)

# confusion matrix and associated statistics for the model fit
## !!!!!!! used for classification not regression

confusionMatrix(data = rfClasses, testingALL$iphonesentiment)
# Confusion Matrix and Statistics

# Reference
# Prediction          very_negative negative somewhat_negative somewhat_positive positive very_positive
# very_negative               386        1                 2                 2        4             5
# negative                      0        1                 1                 0        0             1
# somewhat_negative             2        0                18                 0        0             5
# somewhat_positive             3        0                 2               228        2             7
# positive                      5        0                 3                 2      159            14
# very_positive               192      115               110               124      266          2230

# Overall Statistics

# Accuracy : 0.7769          
# 95% CI : (0.7634, 0.7899)
# No Information Rate : 0.5815 
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.5703          
# Mcnemar's Test P-Value : NA              

# Statistics by Class:

#                           Class: very_negative Class: negative Class: somewhat_negative Class: somewhat_positive
# Sensitivity                       0.65646       0.0085470                 0.132353                  0.64045
# Specificity                       0.99576       0.9994699                 0.998135                  0.99604
# Pos Pred Value                    0.96500       0.3333333                 0.720000                  0.94215
# Neg Pred Value                    0.94212       0.9701569                 0.969470                  0.96491
# Prevalence                        0.15116       0.0300771                 0.034961                  0.09152
# Detection Rate                    0.09923       0.0002571                 0.004627                  0.05861
# Detection Prevalence              0.10283       0.0007712                 0.006427                  0.06221
# Balanced Accuracy                 0.82611       0.5040085                 0.565244                  0.81824

#                       Class: positive Class: very_positive
# Sensitivity                  0.36891               0.9859
# Specificity                  0.99306               0.5043
# Pos Pred Value               0.86885               0.7343
# Neg Pred Value               0.92663               0.9625
# Prevalence                   0.11080               0.5815
# Detection Rate               0.04087               0.5733
# Detection Prevalence         0.04704               0.7807
# Balanced Accuracy            0.68099               0.7451

# var imp  variable importance

rfimp = varImp(rfAll, scale = FALSE)

# summarize importance
print(rfimp)

# rf variable importance

# only 20 most important variables shown (out of 58)

# Overall
# iphone        712.367
# samsunggalaxy 251.948
# iphonedisunc  232.781
# htcphone      214.294
# iphonedisneg  210.796
# googleandroid 196.396
# iphoneperpos  177.166
# iphonedispos  141.013
# iphonecampos  110.004
# iphonecamunc  104.374
# iphoneperneg   92.773
# iphoneperunc   79.627
# iphonecamneg   72.267
# htccampos      38.598
# sonyxperia     30.187
# ios            21.183
# htcdispos      15.610
# samsungperpos  15.435
# sonyperpos     12.248
# htcperpos       9.515

# plot importance
plot(rfimp)

#--- iphoneCORR, correlated features removed ---#

rf_cor <-train(iphonesentiment~.,data=trainingCOR,method="rf",
             trControl= fitControl,
             prox=TRUE,allowParallel=TRUE)

rf_cor

# Random Forest 

# 9083 samples
# 45 predictor
# 6 classes: 'very_negative', 'negative', 'somewhat_negative', 'somewhat_positive', 'positive', 'very_positive' 

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8174, 8175, 8177, 8174, 8173, 8176, ...
# Resampling results across tuning parameters:
  
#   mtry  Accuracy   Kappa   
# 2    0.6908733  0.3434367
# 23    0.7702197  0.5571688
# 45    0.7616541  0.5455627

# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was mtry = 23.

# to predict new samples use predict.train
rfClassesC <- predict(rf_cor, newdata = testingCOR)
str(rfClassesC)
# Factor w/ 6 levels "very_negative",..: 5 6 4 6 6 6 6 6 6 6 ...

postResample(rfClassesC,testingCOR$iphonesentiment)
# Accuracy     Kappa 
# 0.777892 0.573475 

# plot takes the output of the train object and creates a plot
plot(rf_cor)

# confusion matrix and associated statistics for the model fit
## !!!!!!! used for classification not regression

confusionMatrix(data = rfClassesC, testingCOR$iphonesentiment)

# Confusion Matrix and Statistics

#                   Reference
# Prediction          very_negative negative somewhat_negative somewhat_positive positive very_positive
# very_negative               390        0                 2                 2        3             8
# negative                      0        0                 0                 0        0             3
# somewhat_negative             2        1                23                 0        0             3
# somewhat_positive             0        0                 2               238        4            10
# positive                      5        2                 0                 1      151            14
# very_positive               191      114               109               115      273          2224

# Overall Statistics

# Accuracy : 0.7779          
# 95% CI : (0.7645, 0.7909)
# No Information Rate : 0.5815          
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.5735          
# Mcnemar's Test P-Value : NA              


# Statistics by Class:

#              Class: very_negative Class: negative Class: somewhat_negative
# Sensitivity                        0.6633       0.0000000                 0.169118
# Specificity                        0.9955       0.9992049                 0.998402
# Pos Pred Value                     0.9630       0.0000000                 0.793103
# Neg Pred Value                     0.9432       0.9698997                 0.970733
# Prevalence                         0.1512       0.0300771                 0.034961
# Detection Rate                     0.1003       0.0000000                 0.005913
# Detection Prevalence               0.1041       0.0007712                 0.007455
# Balanced Accuracy                  0.8294       0.4996024                 0.583760
# Class: somewhat_positive Class: positive Class: very_positive
# Sensitivity                           0.66854         0.35035               0.9832
# Specificity                           0.99547         0.99364               0.5074
# Pos Pred Value                        0.93701         0.87283               0.7350
# Neg Pred Value                        0.96755         0.92467               0.9560
# Prevalence                            0.09152         0.11080               0.5815
# Detection Rate                        0.06118         0.03882               0.5717
# Detection Prevalence                  0.06530         0.04447               0.7779
# Balanced Accuracy                     0.83201         0.67199               0.7453


# var imp  variable importance

rfimpC = varImp(rf_cor, scale = FALSE)
# summarize importance
print(rfimpC)

# rf variable importance

# only 20 most important variables shown (out of 45)

# Overall
# iphone         742.09
# samsunggalaxy  274.73
# iphonedisunc   251.37
# iphonedisneg   201.60
# iphoneperpos   191.42
# googleandroid  175.17
# iphonedispos   143.60
# iphonecamunc   115.89
# iphonecampos   101.13
# sonyxperia      97.66
# iphoneperneg    93.86
# iphoneperunc    79.79
# iphonecamneg    69.17
# htccampos       66.46
# htcdispos       46.04
# htcperpos       22.93
# samsungperpos   21.33
# htccamneg       13.82
# samsungcamunc   10.91
# htcdisneg       10.71

# plot importance
plot(rfimpC)

#--- iphoneNZV nearly zero variance features removed ---#

rf_nzv <-train(iphonesentiment~.,data=trainingNZV,method="rf",
               trControl= fitControl,
               prox=TRUE,allowParallel=TRUE)
rf_nzv

# Random Forest 

# 9083 samples
# 11 predictor
# 6 classes: 'very_negative', 'negative', 'somewhat_negative', 'somewhat_positive', 'positive', 'very_positive' 

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8176, 8174, 8174, 8172, 8173, 8174, ... 
# Resampling results across tuning parameters:
  
#   mtry  Accuracy   Kappa    
# 2    0.7562810  0.5187863
# 6    0.7549819  0.5216980
# 11    0.7466806  0.5100738

# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was mtry = 2.

# to predict new samples use predict.train
rfClassesnzv <- predict(rf_nzv, newdata = testingNZV)
str(rfClassesnzv)
# Factor w/ 6 levels "very_negative",..: 6 6 6 5 6 6 1 6 6 6 ...

postResample(rfClassesnzv,testingNZV$iphonesentiment)
# Accuracy     Kappa 
# 0.7673522 0.5434449  

# plot(rf_Fit) takes the output of the train object and creates a plot
plot(rf_nzv)

# confusion matrix and associated statistics for the model fit
## !!!!!!! used for classification not regression

confusionMatrix(data = rfClassesnzv, testingNZV$iphonesentiment)
# Confusion Matrix and Statistics

# Reference
# Prediction          very_negative negative somewhat_negative somewhat_positive positive very_positive
# very_negative               401        0                15                 2        2             6
# negative                      0        0                 0                 0        0             0
# somewhat_negative             0        0                 0                 0        0             0
# somewhat_positive             2        0                 0               191        3             2
# positive                      2        0                 0                 4      145             6
# very_positive               183      117               121               159      281          2248

# Overall Statistics

# Accuracy : 0.7674          
# 95% CI : (0.7537, 0.7806)
# No Information Rate : 0.5815  
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.5434          
# Mcnemar's Test P-Value : NA              

# Statistics by Class:

# Class: very_negative Class: negative Class: somewhat_negative
# Sensitivity                        0.6820         0.00000                  0.00000
# Specificity                        0.9924         1.00000                  1.00000
# Pos Pred Value                     0.9413             NaN                      NaN
# Neg Pred Value                     0.9460         0.96992                  0.96504
# Prevalence                         0.1512         0.03008                  0.03496
# Detection Rate                     0.1031         0.00000                  0.00000
# Detection Prevalence               0.1095         0.00000                  0.00000
# Balanced Accuracy                  0.8372         0.50000                  0.50000
# Class: somewhat_positive Class: positive Class: very_positive
# Sensitivity                           0.53652         0.33643               0.9938
# Specificity                           0.99802         0.99653               0.4711
# Pos Pred Value                        0.96465         0.92357               0.7231
# Neg Pred Value                        0.95531         0.92339               0.9821
# Prevalence                            0.09152         0.11080               0.5815
# Detection Rate                        0.04910         0.03728               0.5779
# Detection Prevalence                  0.05090         0.04036               0.7992
# Balanced Accuracy                     0.76727         0.66648               0.7325


rfimpnzv = varImp(rf_nzv, scale = FALSE)

# summarize importance
print(rfimpnzv)

# rf variable importance

# Overall
# iphone         521.50
# htcphone       300.16
# samsunggalaxy  234.44
# iphonedisunc   170.73
# iphonedisneg   148.38
# iphonedispos   119.76
# iphonecamunc   101.05
# iphoneperpos    97.50
# iphonecampos    93.07
# iphoneperneg    69.22
# iphoneperunc    57.82

# plot importance
plot(rfimpnzv)

#--- iphoneRFE  features removed ---#
rf_rfe <- train(iphonesentiment~.,data=trainingRFE,method="rf",
                trControl= fitControl,
                prox=TRUE,allowParallel=TRUE)

rf_rfe

# Random Forest 

# 9083 samples
# 21 predictor
# 6 classes: 'very_negative', 'negative', 'somewhat_negative', 'somewhat_positive', 'positive', 'very_positive' 

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8175, 8174, 8175, 8176, 8175, 8174, ... 
# Resampling results across tuning parameters:
  
#   mtry  Accuracy   Kappa    
# 2    0.7286246  0.4469512
# 11    0.7677505  0.5523206
# 21    0.7608142  0.5431225

# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was mtry = 11.

# to predict new samples use predict.train
rfClassesrfe <- predict(rf_rfe, newdata = testingRFE)
str(rfClassesrfe)
# Factor w/ 6 levels "very_negative",..: 6 6 6 6 6 6 6 6 6 1 ...

postResample(rfClassesrfe,testingRFE$iphonesentiment)
# Accuracy     Kappa 
# 0.7809769 0.5804000  

# plot(rf_Fit) takes the output of the train object and creates a plot
plot(rf_rfe)

# confusion matrix and associated statistics for the model fit
## !!!!!!! used for classification not regression

confusionMatrix(data = rfClassesrfe, testingRFE$iphonesentiment)
# Confusion Matrix and Statistics

# Reference
# Prediction          very_negative negative somewhat_negative somewhat_positive positive very_positive
# very_negative               391        0                 3                 4        2            16
# negative                      0        0                 0                 0        0             3
# somewhat_negative             0        0                21                 0        2             4
# somewhat_positive             3        0                 2               245        8             9
# positive                      2        1                 0                 2      154             3
# very_positive               192      116               110               105      265          2227

# Overall Statistics

# Accuracy : 0.781           
# 95% CI : (0.7676, 0.7939)
# Kappa : 0.5804          
# Mcnemar's Test P-Value : NA              

# Statistics by Class:

# Class: very_negative Class: negative Class: somewhat_negative Class: somewhat_positive
# Sensitivity                        0.6650       0.0000000                 0.154412                  0.68820
# Specificity                        0.9924       0.9992049                 0.998402                  0.99377
# Pos Pred Value                     0.9399       0.0000000                 0.777778                  0.91760
# Neg Pred Value                     0.9433       0.9698997                 0.970230                  0.96936
# Prevalence                         0.1512       0.0300771                 0.034961                  0.09152
# Detection Rate                     0.1005       0.0000000                 0.005398                  0.06298
# Detection Prevalence               0.1069       0.0007712                 0.006941                  0.06864
# Balanced Accuracy                  0.8287       0.4996024                 0.576407                  0.84099
# Class: positive Class: very_positive

# Sensitivity                  0.35731               0.9845
# Specificity                  0.99769               0.5160
# Pos Pred Value               0.95062               0.7386
# Neg Pred Value               0.92570               0.9600
# Prevalence                   0.11080               0.5815
# Detection Rate               0.03959               0.5725
# Detection Prevalence         0.04165               0.7751
# Balanced Accuracy            0.67750               0.7502

# summarize importance
rfimprfe = varImp(rf_rfe, scale = FALSE)
# rf variable importance
print(rfimprfe)


# only 20 most important variables shown (out of 21)

# Overall
# iphone         716.65
# samsunggalaxy  245.82
# htcphone       235.78
# iphonedisunc   234.04
# iphonedisneg   199.59
# iphoneperpos   190.17
# googleandroid  185.97
# iphonedispos   145.63
# iphonecampos   106.35
# iphoneperneg   102.32
# iphonecamunc   100.79
# iphonecamneg    81.66
# iphoneperunc    80.98
# ios             36.40
# htccampos       33.01
# htcdispos       30.64
# samsungperpos   28.85
# sonyxperia      28.12
# htcdisneg       12.23
# htccamneg       11.76

# plot importance
plot(rfimprfe)

#--- iphoneDF, no features removed recoded sentiment target---#

rf_rc <- train(iphonesentiment~.,data=trainingRC,method="rf",
                trControl= fitControl,
                prox=TRUE,allowParallel=TRUE)

rf_rc

# Random Forest 

# 9083 samples
# 58 predictor
# 4 classes: 'negative', 'somewhat_negative', 'somewhat_positive', 'positive' 

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8175, 8175, 8173, 8174, 8175, 8174, ... 
# Resampling results across tuning parameters:
  
#   mtry  Accuracy   Kappa    
# 2    0.7805131  0.3859383
# 30    0.8513715  0.6321344
# 58    0.8458006  0.6216072

# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was mtry = 30.

# to predict new samples use predict.train
rfClassesrc <- predict(rf_rc, newdata = testingRC)
str(rfClassesrc)
# Factor w/ 4 levels "negative","somewhat_negative",..: 4 4 4 1 4 4 4 4 4 4 ...

postResample(rfClassesrc,testingRC$iphonesentiment)
# Accuracy     Kappa 
# 0.8452442 0.6113755 

# plot(rf_Fit) takes the output of the train object and creates a plot
plot(rf_rc)

# confusion matrix and associated statistics for the model fit
## !!!!!!! used for classification not regression
confusionMatrix(data = rfClassesrc, testingRC$iphonesentiment)
# Confusion Matrix and Statistics

# Reference
# Prediction          negative somewhat_negative somewhat_positive positive
# negative               367                 2                 4        6
# somewhat_negative        1                16                 0        0
# somewhat_positive        3                 2               229       11
# positive               334               116               123     2676

# Overall Statistics

# Accuracy : 0.8452          
# 95% CI : (0.8335, 0.8565)
# No Information Rate : 0.6923          
# P-Value [Acc > NIR] : < 2.2e-16   

# Kappa : 0.6114          
# Mcnemar's Test P-Value : < 2.2e-16       

# Statistics by Class:

# Class: negative Class: somewhat_negative
# Sensitivity                  0.52057                 0.117647
# Specificity                  0.99623                 0.999734
# Pos Pred Value               0.96834                 0.941176
# Neg Pred Value               0.90373                 0.969016
# Prevalence                   0.18123                 0.034961
# Detection Rate               0.09434                 0.004113
# Detection Prevalence         0.09743                 0.004370
# Balanced Accuracy            0.75840                 0.558690
# Class: somewhat_positive Class: positive

# Sensitivity                           0.64326          0.9937
# Specificity                           0.99547          0.5213
# Pos Pred Value                        0.93469          0.8236
# Neg Pred Value                        0.96516          0.9735
# Prevalence                            0.09152          0.6923
# Detection Rate                        0.05887          0.6879
# Detection Prevalence                  0.06298          0.8352
# Balanced Accuracy                     0.81937          0.7575

# summarize importance
rfimprc = varImp(rf_rc, scale = FALSE)

# rf variable importance
print(rfimprc)
# rf variable importance

# only 20 most important variables shown (out of 58)

# Overall
# iphone         722.98
# samsunggalaxy  277.31
# htcphone       233.83
# googleandroid  222.72
# iphoneperpos   161.91
# iphonedisneg   127.68
# iphonedispos   112.68
# iphonedisunc   102.97
# iphoneperneg    70.30
# iphoneperunc    62.67
# iphonecampos    60.13
# sonyxperia      34.88
# iphonecamunc    33.17
# htccampos       31.39
# iphonecamneg    29.32
# htcdispos       24.58
# sonyperpos      14.83
# samsungperpos   13.50
# ios             13.49
# samsungdispos   12.41

# plot importance
plot(rfimprc)

#--- iphone, Principal Component Analysis PCA ---#
rf_pca <-  train(iphonesentiment~.,data=train.pca,method="rf",
                 trControl= fitControl,
                 prox=TRUE,allowParallel=TRUE)

rf_pca  

# Random Forest 

# 9083 samples
# 25 predictor
# 6 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8174, 8174, 8175, 8174, 8173, 8176, ... 
# Resampling results across tuning parameters:
  
#   mtry  Accuracy   Kappa    
# 2    0.7594197  0.5370281
# 13    0.7597721  0.5383356
# 25    0.7591886  0.5374566

# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was mtry = 13.

rfClassespca <- predict(rf_pca, newdata = test.pca)
str(rfClassespca)
# Factor w/ 6 levels "very_negative",..: 5 6 6 6 6 6 6 6 6 6 ...

postResample(rfClassespca,test.pca$iphonesentiment)
# Accuracy     Kappa 
# 0.7642674 0.5487185

# Using the option type = "prob" can be used to compute class
# probabilities from the model. 

rfProbspca <- predict(rf_pca, newdata = test.pca, type = "prob")

head(rfProbspca)

# very_negative negative somewhat_negative somewhat_positive positive very_positive
# 7          0.000    0.000             0.000             0.000    1.000         0.000
# 8          0.002    0.004             0.004             0.004    0.002         0.984
# 17         0.000    0.000             0.000             0.000    0.000         1.000
# 30         0.000    0.000             0.000             0.000    0.000         1.000
# 33         0.032    0.006             0.000             0.040    0.012         0.910
# 34         0.002    0.004             0.002             0.008    0.004         0.980

plot(rf_pca)

# confusion matrix and associated statistics for the model fit
## !!!!!!! used for classification not regression 

confusionMatrix(data = rfClassespca, test.pca$iphonesentiment)
# Confusion Matrix and Statistics

# Reference
# Prediction          very_negative negative somewhat_negative somewhat_positive positive very_positive
# very_negative               387        1                 3                13       16            21
# negative                      1        1                 2                 0        1             3
# somewhat_negative             3        0                18                 0        0             6
# somewhat_positive             1        0                 2               218        0             8
# positive                      4        2                 1                 2      146            21
# very_positive               192      113               110               123      268          2203

# Overall Statistics

# Accuracy : 0.7643          
# 95% CI : (0.7506, 0.7775)
# No Information Rate : 0.5815          
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.5487          
# Mcnemar's Test P-Value : NA 
# Statistics by Class:

# Class: very_negative Class: negative Class: somewhat_negative Class: somewhat_positive
# Sensitivity                       0.65816       0.0085470                 0.132353                  0.61236
# Specificity                       0.98365       0.9981447                 0.997603                  0.99689
# Pos Pred Value                    0.87755       0.1250000                 0.666667                  0.95197
# Neg Pred Value                    0.94172       0.9701185                 0.969454                  0.96231
# Prevalence                        0.15116       0.0300771                 0.034961                  0.09152
# Detection Rate                    0.09949       0.0002571                 0.004627                  0.05604
# Detection Prevalence              0.11337       0.0020566                 0.006941                  0.05887
# Balanced Accuracy                 0.82090       0.5033459                 0.564978                  0.80462
# Class: positive Class: very_positive
# Sensitivity                  0.33875               0.9739
# Specificity                  0.99133               0.5049
# Pos Pred Value               0.82955               0.7321
# Neg Pred Value               0.92326               0.9330
# Prevalence                   0.11080               0.5815
# Detection Rate               0.03753               0.5663
# Detection Prevalence         0.04524               0.7735
# Balanced Accuracy            0.66504               0.7394

# var imp
rfimppca = varImp(rf_pca, scale = FALSE)

# plot importance
rfimppca

# rf variable importance

# only 20 most important variables shown (out of 25)

# Overall
# PC5   553.07
# PC11  265.00
# PC4   168.93
# PC3   155.50
# PC1   151.59
# PC12  146.29
# PC10  132.39
# PC14  124.60
# PC16  108.45
# PC2   104.72
# PC17   99.70
# PC22   89.65
# PC6    85.02
# PC13   84.48
# PC24   83.27
# PC25   74.61
# PC7    73.21
# PC9    68.47
# PC20   67.83
# PC15   56.88

plot(rfimppca)

#--- k-Nearest Neighbors   ---#
modelLookup("kknn")

#--- iphoneDF, no features removed ---#

knn_ALLa <- train(iphonesentiment ~., data = trainingALL, method = "kknn",
                  trControl=fitControl,
                  na.action = na.omit,
               #  kmax = 9)
                  preProcess = c("center", "scale"))
                # tuneLength = 10)

knn_ALLa

# k-Nearest Neighbors 

# 9083 samples
# 58 predictor
# 6 classes: 'very_negative', 'negative', 'somewhat_negative', 'somewhat_positive', 'positive', 'very_positive' 

# Pre-processing: centered (58), scaled (58) 
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8175, 8176, 8175, 8176, 8174, 8174, ... 
# Resampling results across tuning parameters:
# kmax  Accuracy   Kappa    
#  5     0.3098552  0.1516097
#  7     0.3222295  0.1573219
#  9     0.3303881  0.1620847

# Tuning parameter 'distance' was held constant at a value of 2
# Tuning parameter 'kernel' was held constant at a
# value of optimal
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were kmax = 9, distance = 2 and kernel = optimal.

#####  Other performance metrics not run for knn_ALLa, only done below for knn_ALL
#####  This is because of low accuracy and kappa for both

knn_ALL <- train(iphonesentiment ~., data = trainingALL, method = "kknn",
                  trControl=fitControl,
                  na.action = na.omit,
                  #  kmax = 9)
                  preProcess = c("center", "scale"),
                  tuneLength = 10)

knn_ALL


# k-Nearest Neighbors 

# 9083 samples
# 58 predictor
# 6 classes: 'very_negative', 'negative', 'somewhat_negative', 'somewhat_positive', 'positive', 'very_positive' 

# Pre-processing: centered (58), scaled (58) 
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8175, 8176, 8174, 8175, 8176, 8173, ... 
# Resampling results across tuning parameters:
#   kmax  Accuracy   Kappa    
# 5    0.3100058  0.1517274
# 7    0.3229532  0.1578144
# 9    0.3300327  0.1615918
# 11    0.3381136  0.1659767
# 13    0.3415273  0.1663402
# 15    0.3504563  0.1720976
# 17    0.3554214  0.1750156
# 19    0.3591099  0.1768353
# 21    0.3625559  0.1787922
# 23    0.3677853  0.1823216

# Tuning parameter 'distance' was held constant at a value of 2
# Tuning parameter
# 'kernel' was held constant at a value of optimal
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were kmax = 23, distance = 2 and kernel = optimal.

# to predict new samples use predict.train
knnClasses <- predict(knn_ALL, newdata = testingALL)
str(knnClasses)
# Factor w/ 6 levels "very_negative",..: 5 6 1 1 6 6 1 1 1 1 ...

postResample(knnClasses,testingALL$iphonesentiment)
# Accuracy     Kappa 
# 0.3609254 0.1818239

# plot(rf_Fit) takes the output of the train object and creates a plot
plot(knn_ALL)

# confusion matrix and associated statistics for the model fit
## !!!!!!! used for classification not regression

confusionMatrix(data = knnClasses, testingALL$iphonesentiment)

# Confusion Matrix and Statistics

# Reference
# Prediction          very_negative negative somewhat_negative somewhat_positive positive very_positive
# very_negative               525       84                77               106      209          1664
# negative                      3        1                 2                 2        2            32
# somewhat_negative             6        1                18                 1        2            20
# somewhat_positive             9        0                 5               224        5            29
# positive                      2        4                 3                 2      140            21
# very_positive                43       27                31                21       73           496

# Overall Statistics

# Accuracy : 0.3609          
# 95% CI : (0.3458, 0.3762)
# No Information Rate : 0.5815          
# P-Value [Acc > NIR] : 1  
# Kappa : 0.1818          
# Mcnemar's Test P-Value : <2e-16          

# Statistics by Class:

# Class: very_negative Class: negative Class: somewhat_negative Class: somewhat_positive
# Sensitivity                        0.8929       0.0085470                 0.132353                  0.62921
# Specificity                        0.3519       0.9891333                 0.992009                  0.98642
# Pos Pred Value                     0.1970       0.0238095                 0.375000                  0.82353
# Neg Pred Value                     0.9486       0.9698545                 0.969287                  0.96352
# Prevalence                         0.1512       0.0300771                 0.034961                  0.09152
# Detection Rate                     0.1350       0.0002571                 0.004627                  0.05758
# Detection Prevalence               0.6851       0.0107969                 0.012339                  0.06992
# Balanced Accuracy                  0.6224       0.4988402                 0.562181                  0.80782
# Class: positive Class: very_positive
# Sensitivity                  0.32483               0.2193
# Specificity                  0.99075               0.8802
# Pos Pred Value               0.81395               0.7178
# Neg Pred Value               0.92173               0.4480
# Prevalence                   0.11080               0.5815
# Detection Rate               0.03599               0.1275
# Detection Prevalence         0.04422               0.1776
# Balanced Accuracy            0.65779               0.5497

knnimp = varImp(knn_ALL, scale = FALSE)

# summarize importance
print(knnimp)
# ROC curve variable importance

# variables are sorted by maximum importance across the classes
# only 20 most important variables shown (out of 58)

# very_negative negative somewhat_negative somewhat_positive positive very_positive
# iphone               0.6581   0.6581            0.7080            0.6581   0.6581        0.6552
# htcphone             0.7070   0.7055            0.7055            0.7067   0.7055        0.7070
# iphonedisunc         0.5084   0.5834            0.6689            0.5084   0.5084        0.5834
# iphonedisneg         0.5030   0.5163            0.6564            0.5030   0.5055        0.5163
# samsunggalaxy        0.6508   0.6503            0.6503            0.6506   0.6503        0.6508
# iphonedispos         0.5149   0.5926            0.6387            0.5026   0.5131        0.5926
# iphonecamunc         0.5439   0.5439            0.6309            0.5439   0.5439        0.5388
# iphonecamneg         0.5389   0.5373            0.6190            0.5373   0.5373        0.5389
# iphonecampos         0.5512   0.5512            0.6032            0.5512   0.5512        0.5470
# htccampos            0.5983   0.5965            0.5965            0.5975   0.5965        0.5983
# htcdispos            0.5948   0.5944            0.5944            0.5966   0.5944        0.5948
# htcperpos            0.5957   0.5925            0.5925            0.5937   0.5925        0.5957
# htcperneg            0.5826   0.5810            0.5810            0.5816   0.5810        0.5826
# htcdisneg            0.5794   0.5792            0.5792            0.5799   0.5792        0.5794
# htccamneg            0.5753   0.5753            0.5753            0.5753   0.5753        0.5753
# htcdisunc            0.5753   0.5753            0.5753            0.5753   0.5753        0.5753
# htcperunc            0.5731   0.5731            0.5731            0.5731   0.5731        0.5731
# googleandroid        0.5340   0.5340            0.5340            0.5354   0.5721        0.5332
# htccamunc            0.5702   0.5702            0.5702            0.5702   0.5702        0.5702
# iphoneperpos         0.5180   0.5283            0.5694            0.5256   0.5161        0.5283

# plot importance
plot(knnimp)

#--- Support Vector Machine   ---#
modelLookup("svmLinear2")

#--- iphoneDF, no features removed ---#

svm_ALL <- train(iphonesentiment ~., data = trainingALL, method = "svmLinear2",
                 trControl=fitControl,
                 na.action = na.omit,
                 preProcess = c("center", "scale"))

                 #,tuneLength = 10)
              

svm_ALL
# Support Vector Machines with Linear Kernel 

# 9083 samples
# 58 predictor
# 6 classes: 'very_negative', 'negative', 'somewhat_negative', 'somewhat_positive', 'positive', 'very_positive' 

# Pre-processing: centered (58), scaled (58) 
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8176, 8174, 8175, 8175, 8175, 8174, ... 
# Resampling results across tuning parameters:
  
#   cost  Accuracy   Kappa    
#  0.25  0.7023353  0.3965100
# 0.50  0.7067169  0.4062418
# 1.00  0.7109988  0.4160263

# Accuracy was used to select the optimal model using the largest value.
# # The final value used for the model was cost = 1.

# to predict new samples use predict.train
svmClasses <- predict(svm_ALL, newdata = testingALL)
str(svmClasses)
# Factor w/ 6 levels "very_negative",..: 6 6 6 6 6 6 6 6 6 6 ...

postResample(svmClasses,testingALL$iphonesentiment)
# Accuracy     Kappa 
# 0.7146530 0.4273227 

# plot(rf_Fit) takes the output of the train object and creates a plot
plot(svm_ALL)

# confusion matrix and associated statistics for the model fit
## !!!!!!! used for classification not regression

confusionMatrix(data = svmClasses, testingALL$iphonesentiment)
# Confusion Matrix and Statistics

# Reference
# Prediction          very_negative negative somewhat_negative somewhat_positive positive very_positive
# very_negative               374        1                 3                22       13            41
# negative                      0        0                 0                 0        0             0
# somewhat_negative             2        0                 2                 0        0             0
# somewhat_positive             5        0                16               103        2             7
# positive                      1        0                 1                 0       93             6
# very_positive               206      116               114               231      323          2208

# Overall Statistics

# Accuracy : 0.7147          
# 95% CI : (0.7002, 0.7288)
# No Information Rate : 0.5815          
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.4273          
# Mcnemar's Test P-Value : NA              

# Statistics by Class:
#                Class: very_negative Class: negative Class: somewhat_negative
# Sensitivity                       0.63605         0.00000                0.0147059
# Specificity                       0.97577         1.00000                0.9994672
# Pos Pred Value                    0.82379             NaN                0.5000000
# Neg Pred Value                    0.93772         0.96992                0.9655172
# Prevalence                        0.15116         0.03008                0.0349614
# Detection Rate                    0.09614         0.00000                0.0005141
# Detection Prevalence              0.11671         0.00000                0.0010283
# Balanced Accuracy                 0.80591         0.50000                0.5070866
# Class: somewhat_positive Class: positive Class: very_positive
# Sensitivity                           0.28933         0.21578               0.9761
# Specificity                           0.99151         0.99769               0.3919
# Pos Pred Value                        0.77444         0.92079               0.6904
# Neg Pred Value                        0.93266         0.91079               0.9220
# Prevalence                            0.09152         0.11080               0.5815
# Detection Rate                        0.02648         0.02391               0.5676
# Detection Prevalence                  0.03419         0.02596               0.8221
# Balanced Accuracy                     0.64042         0.60673               0.6840

svmimp = varImp(svm_ALL, scale = FALSE)

# summarize importance
# print(svmimp)
# ROC curve variable importance

# variables are sorted by maximum importance across the classes
# only 20 most important variables shown (out of 58)

# very_negative negative somewhat_negative somewhat_positive positive very_positive
# iphone               0.6581   0.6581            0.7080            0.6581   0.6581        0.6552
# htcphone             0.7070   0.7055            0.7055            0.7067   0.7055        0.7070
# iphonedisunc         0.5084   0.5834            0.6689            0.5084   0.5084        0.5834
# iphonedisneg         0.5030   0.5163            0.6564            0.5030   0.5055        0.5163
# samsunggalaxy        0.6508   0.6503            0.6503            0.6506   0.6503        0.6508
# iphonedispos         0.5149   0.5926            0.6387            0.5026   0.5131        0.5926
# iphonecamunc         0.5439   0.5439            0.6309            0.5439   0.5439        0.5388
# iphonecamneg         0.5389   0.5373            0.6190            0.5373   0.5373        0.5389
# iphonecampos         0.5512   0.5512            0.6032            0.5512   0.5512        0.5470
# htccampos            0.5983   0.5965            0.5965            0.5975   0.5965        0.5983
# htcdispos            0.5948   0.5944            0.5944            0.5966   0.5944        0.5948
# htcperpos            0.5957   0.5925            0.5925            0.5937   0.5925        0.5957
# htcperneg            0.5826   0.5810            0.5810            0.5816   0.5810        0.5826
# htcdisneg            0.5794   0.5792            0.5792            0.5799   0.5792        0.5794
# htccamneg            0.5753   0.5753            0.5753            0.5753   0.5753        0.5753
# htcdisunc            0.5753   0.5753            0.5753            0.5753   0.5753        0.5753
# htcperunc            0.5731   0.5731            0.5731            0.5731   0.5731        0.5731
# googleandroid        0.5340   0.5340            0.5340            0.5354   0.5721        0.5332
# htccamunc            0.5702   0.5702            0.5702            0.5702   0.5702        0.5702
# iphoneperpos         0.5180   0.5283            0.5694            0.5256   0.5161        0.5283

plot(svmimp)

################################
# Train model for samsung galaxy
################################

## ------- C5.0 ------- ##

modelLookup('C5.0')



## ------- samsungDF  ------- ##

c5_allSam <-  train(galaxysentiment ~ .,
                 data = trainingALLsam,
                 method = "C5.0",
                 tuneLength = 2,
                 trControl = fitControl,
                 # na.action = na.pass)
                 na.action = na.omit)

c5_allSam

# C5.0 

# 9040 samples
# 58 predictor
# 6 classes: 'very_negative', 'negative', 'somewhat_negative', 'somewhat_positive', 'positive', 'very_positive' 

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8135, 8137, 8137, 8134, 8137, 8136, ... 
# Resampling results across tuning parameters:
  
#   model  winnow  trials  Accuracy   Kappa    
# rules  FALSE    1      0.7687487  0.5379672
# rules  FALSE   10      0.7642571  0.5293282
# rules   TRUE    1      0.7684167  0.5376988
# rules   TRUE   10      0.7630177  0.5263439
# tree   FALSE    1      0.7678970  0.5370000
# tree   FALSE   10      0.7637484  0.5307199
# tree    TRUE    1      0.7678305  0.5367648
# tree    TRUE   10      0.7628083  0.5286026

# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = rules and winnow = FALSE.


c5ClassesSam <- predict(c5_allSam, newdata = testingALLsam)
str(c5ClassesSam)
# Factor w/ 6 levels "very_negative",..: 6 1 6 4 6 6 6 6 6 6 ...

postResample(c5ClassesSam,testingALLsam$galaxysentiment)
# Accuracy     Kappa 
# 0.7576854 0.5092403 

# plot takes the output of the train object and creates a plot
plot(c5_allSam)

confusionMatrix(data = c5ClassesSam, testingALLsam$galaxysentiment)

# Confusion Matrix and Statistics

# Reference
# Prediction          very_negative negative somewhat_negative somewhat_positive positive very_positive
# very_negative               329        2                 0                 1        5            22
# negative                      0        0                 0                 0        0             0
# somewhat_negative             1        0                15                 1        0             1
# somewhat_positive            10        3                 1               204        7            31
# positive                      8        0                 1                 1      113            11
# very_positive               160      109               118               145      300          2272

# Overall Statistics

# Accuracy : 0.7577          
# 95% CI : (0.7439, 0.7711)
# No Information Rate : 0.6037          
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.5092          
# Mcnemar's Test P-Value : NA              

# Statistics by Class:
#                     Class: very_negative Class: negative Class: somewhat_negative Class: somewhat_positive
# Sensitivity                       0.64764         0.00000                 0.111111                  0.57955
# Specificity                       0.99108         1.00000                 0.999197                  0.98522
# Pos Pred Value                    0.91643             NaN                 0.833333                  0.79687
# Neg Pred Value                    0.94903         0.97055                 0.968855                  0.95906
# Prevalence                        0.13123         0.02945                 0.034875                  0.09093
# Detection Rate                    0.08499         0.00000                 0.003875                  0.05270
# Detection Prevalence              0.09274         0.00000                 0.004650                  0.06613
# Balanced Accuracy                 0.81936         0.50000                 0.555154                  0.78238
# Class: positive Class: very_positive
# Sensitivity                  0.26588               0.9722
# Specificity                  0.99391               0.4576
# Pos Pred Value               0.84328               0.7320
# Neg Pred Value               0.91651               0.9153
# Prevalence                   0.10979               0.6037
# Detection Rate               0.02919               0.5869
# Detection Prevalence         0.03462               0.8019
# Balanced Accuracy            0.62989               0.7149

c5impSam = varImp(c5_allSam, scale = FALSE)
print(c5impSam)
# C5.0 variable importance

# only 20 most important variables shown (out of 58)

# Overall
# iphone          99.89
# samsungdispos   98.60
# samsungcamunc   98.53
# iphonedisneg    10.24
# htccampos        9.96
# iphoneperpos     9.94
# googleandroid    9.47
# iphonedispos     7.73
# iphonecamneg     7.46
# samsungperneg    7.10
# iphonecampos     6.27
# sonyxperia       4.93
# iphonedisunc     4.50
# samsunggalaxy    3.94
# samsungcamneg    2.74
# iphoneperunc     2.07
# ios              1.98
# iphoneperneg     1.70
# htcphone         1.06
# htcdisunc        0.13

plot(c5impSam)

#--- samsungCOR, correlated features removed ---#

c5_corSam <-  train(galaxysentiment ~ .,
                 data = trainingCORsam,
                 method = "C5.0",
                 tuneLength = 2,
                 trControl = fitControl,
                 # na.action = na.pass)
                 na.action = na.omit)

c5_corSam
# C5.0 

# 9040 samples
# 44 predictor
# 6 classes: 'very_negative', 'negative', 'somewhat_negative', 'somewhat_positive', 'positive', 'very_positive' 

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8133, 8136, 8137, 8137, 8135, 8137, ... 
# Resampling results across tuning parameters:
  
#   model  winnow  trials  Accuracy   Kappa    
# rules  FALSE    1      0.7644465  0.5255351
# rules  FALSE   10      0.7560948  0.5098118
# rules   TRUE    1      0.7639260  0.5244624
# rules   TRUE   10      0.7559290  0.5091682
# tree   FALSE    1      0.7634505  0.5246110
# tree   FALSE   10      0.7551649  0.5091271
# tree    TRUE    1      0.7632399  0.5243568
# tree    TRUE   10      0.7549214  0.5086431

# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = rules and winnow = FALSE.

c5ClassesCORSam <- predict(c5_corSam, newdata = testingCORsam)
str(c5ClassesCORSam)
# Factor w/ 6 levels "very_negative",..: 6 6 6 6 4 1 6 6 4 6 ...
postResample(c5ClassesCORSam,testingCORsam$galaxysentiment)
# Accuracy     Kappa 
# 0.7708602 0.5427929 

# plottakes the output of the train object and creates a plot
plot(c5_corSam)
confusionMatrix(data = c5ClassesCORSam, testingCORsam$galaxysentiment)
# Confusion Matrix and Statistics

# Reference
# Prediction          very_negative negative somewhat_negative somewhat_positive positive very_positive
# very_negative               363        1                 2                 7       10            31
# negative                      0        0                 0                 0        0             0
# somewhat_negative             0        0                22                 0        0             2
# somewhat_positive             3        0                 1               211        7            24
# positive                      5        0                 1                 1      122            14
# very_positive               137      113               109               133      286          2266

# Overall Statistics

# Accuracy : 0.7709         
# 95% CI : (0.7573, 0.784)
# No Information Rate : 0.6037         
# P-Value [Acc > NIR] : < 2.2e-16      

# Kappa : 0.5428         
# Mcnemar's Test P-Value : NA   
# Statistics by Class:

# Class: very_negative Class: negative Class: somewhat_negative Class: somewhat_positive
# Sensitivity                       0.71457         0.00000                 0.162963                  0.59943
# Specificity                       0.98483         1.00000                 0.999465                  0.99005
# Pos Pred Value                    0.87681             NaN                 0.916667                  0.85772
# Neg Pred Value                    0.95806         0.97055                 0.970626                  0.96110
# Prevalence                        0.13123         0.02945                 0.034875                  0.09093
# Detection Rate                    0.09377         0.00000                 0.005683                  0.05451
# Detection Prevalence              0.10695         0.00000                 0.006200                  0.06355
# Balanced Accuracy                 0.84970         0.50000                 0.581214                  0.79474
# Class: positive Class: very_positive
# Sensitivity                  0.28706               0.9696
# Specificity                  0.99391               0.4928
# Pos Pred Value               0.85315               0.7444
# Neg Pred Value               0.91872               0.9141
# Prevalence                   0.10979               0.6037
# Detection Rate               0.03152               0.5854
# Detection Prevalence         0.03694               0.7864
# Balanced Accuracy            0.64048               0.7312

c5impCORSam = varImp(c5_corSam, scale = FALSE)
print(c5impCORSam)

# C5.0 variable importance

# only 20 most important variables shown (out of 44)

# Overall
# iphone         100.00
# googleandroid    9.14
# iphonedisneg     8.84
# iphonedispos     7.61
# htccampos        7.16
# iphoneperpos     7.06
# iphonecampos     5.59
# sonyxperia       5.33
# iphonedisunc     4.34
# samsunggalaxy    3.48
# iphoneperunc     3.02
# samsungcamneg    2.78
# iphoneperneg     2.11
# htcperpos        1.98
# iosperpos        1.79
# iphonecamneg     1.70
# htcdispos        1.01
# htcdisunc        0.00
# samsungcampos    0.00
# sonyperneg       0.00

plot(c5impCORSam)

#--- samsungNZV nearly zero variance features removed ---#

c5_nzvSam <-  train(galaxysentiment ~ .,
                 data = trainingNZVsam,
                 method = "C5.0",
                 tuneLength = 2,
                 trControl = fitControl,
                 # na.action = na.pass)
                 na.action = na.omit)

c5_nzvSam
# C5.0 

# 9040 samples
# 11 predictor
# 6 classes: 'very_negative', 'negative', 'somewhat_negative', 'somewhat_positive', 'positive', 'very_positive' 

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8136, 8135, 8137, 8135, 8134, 8138, ... 
# Resampling results across tuning parameters:
  
  
#   model  winnow  trials  Accuracy   Kappa    
# rules  FALSE    1      0.7525334  0.4985842
# rules  FALSE   10      0.7450657  0.4801415
# rules   TRUE    1      0.7524447  0.4984901
# rules   TRUE   10      0.7436601  0.4765091
# tree   FALSE    1      0.7517365  0.4977321
# tree   FALSE   10      0.7449228  0.4815643
# tree    TRUE    1      0.7517586  0.4978585
# tree    TRUE   10      0.7432629  0.4777680

# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = rules and winnow = FALSE.

c5ClassesNZVsam <- predict(c5_nzvSam, newdata = testingNZVsam)
str(c5ClassesNZVsam)
# Factor w/ 6 levels "very_negative",..: 4 6 6 6 4 4 6 4 6 6 ...
postResample(c5ClassesNZVsam,testingNZVsam$galaxysentiment)
# Accuracy     Kappa 
#  0.7543270 0.4992808 

# plot takes the output of the train object and creates a plot
plot(c5_nzvSam)
confusionMatrix(data = c5ClassesNZVsam, testingNZVsam$galaxysentiment)
# Confusion Matrix and Statistics

# Reference
# Prediction          very_negative negative somewhat_negative somewhat_positive positive very_positive
# very_negative               353        0                16                 0        9            28
# negative                      0        0                 0                 0        0             0
# somewhat_negative             0        0                 0                 0        0             0
# somewhat_positive             1        0                 0               162        5            19
# positive                      4        2                 1                 1      132            17
# very_positive               150      112               118               189      279          2273

# Overall Statistics

# Accuracy : 0.7543          
# 95% CI : (0.7404, 0.7678)
# No Information Rate : 0.6037          
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.4993          
# Mcnemar's Test P-Value : NA   
# Statistics by Class:

# Class: very_negative Class: negative Class: somewhat_negative Class: somewhat_positive Class: positive
# Sensitivity                       0.69488         0.00000                  0.00000                  0.46023         0.31059
# Specificity                       0.98424         1.00000                  1.00000                  0.99290         0.99275
# Pos Pred Value                    0.86946             NaN                      NaN                  0.86631         0.84076
# Neg Pred Value                    0.95527         0.97055                  0.96513                  0.94843         0.92111
# Prevalence                        0.13123         0.02945                  0.03487                  0.09093         0.10979
# Detection Rate                    0.09119         0.00000                  0.00000                  0.04185         0.03410
# Detection Prevalence              0.10488         0.00000                  0.00000                  0.04831         0.04056
# Balanced Accuracy                 0.83956         0.50000                  0.50000                  0.72656         0.65167
# Class: very_positive
# Sensitivity                        0.9726
# Specificity                        0.4472
# Pos Pred Value                     0.7283
# Neg Pred Value                     0.9147
# Prevalence                         0.6037
# Detection Rate                     0.5872
# Detection Prevalence               0.8063
# Balanced Accuracy                  0.7099

c5impNZVsam = varImp(c5_nzvSam, scale = FALSE)
print(c5impNZVsam)

# C5.0 variable importance

# Overall
# iphone         100.00
# htcphone        10.70
# samsunggalaxy   10.00
# iphoneperpos     9.97
# iphonedisneg     8.61
# iphonecamunc     6.79
# iphonecampos     6.31
# iphonedispos     6.15
# iphonedisunc     5.44
# iphoneperunc     5.15
# iphoneperneg     4.17

plot(c5impNZVsam)

#--- samsungRFE  features removed ---#

c5_rfeSam <-  train(galaxysentiment ~ .,
                 data = trainingRFEsam,
                 method = "C5.0",
                 tuneLength = 2,
                 trControl = fitControl,
                 # na.action = na.pass)
                 na.action = na.omit)

c5_rfeSam

# 5.0 

# 9040 samples
# 11 predictor
# 6 classes: 'very_negative', 'negative', 'somewhat_negative', 'somewhat_positive', 'positive', 'very_positive' 

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8135, 8136, 8138, 8136, 8135, 8136, ... 
# Resampling results across tuning parameters:
  
#   model  winnow  trials  Accuracy   Kappa    
# rules  FALSE    1      0.7508180  0.5004469
# rules  FALSE   10      0.7445573  0.4812194
# rules   TRUE    1      0.7507516  0.4998865
# rules   TRUE   10      0.7446901  0.4805643
# tree   FALSE    1      0.7510284  0.5018850
# tree   FALSE   10      0.7420352  0.4780928
# tree    TRUE    1      0.7508404  0.5011186
# tree    TRUE   10      0.7415931  0.4758776

# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = tree and winnow = FALSE.

c5ClassesRFEsam <- predict(c5_rfeSam, newdata = testingRFEsam)
str(c5ClassesRFEsam)
# Factor w/ 6 levels "very_negative",..: 6 1 6 4 4 6 6 4 1 6 ...
postResample(c5ClassesRFEsam,testingRFEsam$galaxysentiment)
# Accuracy     Kappa 
# 0.7576854 0.5179587 

# plot takes the output of the train object and creates a plot
plot(c5_rfeSam)

confusionMatrix(data = c5ClassesRFEsam, testingRFEsam$galaxysentiment)

# Confusion Matrix and Statistics

# Reference
# Prediction          very_negative negative somewhat_negative somewhat_positive positive very_positive
# very_negative               352        0                 1                 8       13            34
# negative                      0        0                 0                 0        0             0
# somewhat_negative             1        0                16                 0        0             1# 
# somewhat_positive             3        1                 5               194        8            48
# positive                      3        1                 2                 1      132            15
# very_positive               149      112               111               149      272          2239

# Overall Statistics

# Accuracy : 0.7577          
# 95% CI : (0.7439, 0.7711)
# No Information Rate : 0.6037          
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.518           
# Mcnemar's Test P-Value : NA    

# Statistics by Class:
  
#   Class: very_negative Class: negative Class: somewhat_negative Class: somewhat_positive Class: positive
# Sensitivity                       0.69291         0.00000                 0.118519                  0.55114         0.31059
# Specificity                       0.98335         1.00000                 0.999465                  0.98153         0.99362
# Pos Pred Value                    0.86275             NaN                 0.888889                  0.74903         0.85714
# Neg Pred Value                    0.95495         0.97055                 0.969115                  0.95626         0.92117
# Prevalence                        0.13123         0.02945                 0.034875                  0.09093         0.10979
# Detection Rate                    0.09093         0.00000                 0.004133                  0.05012         0.03410
# Detection Prevalence              0.10540         0.00000                 0.004650                  0.06691         0.03978
# Balanced Accuracy                 0.83813         0.50000                 0.558992                  0.76633         0.65210
# Class: very_positive
# Sensitivity                        0.9581
# Specificity                        0.4831
# Pos Pred Value                     0.7385
# Neg Pred Value                     0.8832
# Prevalence                         0.6037
# Detection Rate                     0.5784
# Detection Prevalence               0.7833
# Balanced Accuracy                  0.7206

c5impRFEsam = varImp(c5_rfeSam, scale = FALSE)
print(c5impRFEsam)

# C5.0 variable importance

# Overall
# iphone         100.00
# googleandroid   96.28
# htccampos       92.53
# sonyxperia      90.35
# samsunggalaxy   89.71
# iphoneperpos    89.34
# iphonedisunc    83.96
# iphonedispos    77.41
# iphonecamneg    15.80
# iphonecampos     6.28
# htcphone         2.43

plot(c5impRFEsam)

#--- samsungRC, no features removed recoded sentiment target---#
c5_rcSam <-  train(galaxysentiment ~ .,
                data = trainingRCSam,
                method = "C5.0",
                tuneLength = 2,
                trControl = fitControl,
                # na.action = na.pass)
                na.action = na.omit)

c5_rcSam

# C5.0 

# 9039 samples
# 58 predictor
# 4 classes: 'negative', 'somewhat_negative', 'somewhat_positive', 'positive' 

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8134, 8134, 8136, 8136, 8135, 8136, ... 
# Resampling results across tuning parameters:
  
#   model  winnow  trials  Accuracy   Kappa    
# rules  FALSE    1      0.8395842  0.5804914
# rules  FALSE   10      0.8398280  0.5816485
# rules   TRUE    1      0.8396177  0.5807347
# rules   TRUE   10      0.8385002  0.5773227
# tree   FALSE    1      0.8386326  0.5798741
# tree   FALSE   10      0.8389424  0.5812033
# tree    TRUE    1      0.8390864  0.5808787
# tree    TRUE   10      0.8368520  0.5757464

# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 10, model = rules and winnow = FALSE.  
  
c5ClassesRCSam <- predict(c5_rcSam, newdata = testingRCSam)
str(c5ClassesRCSam)
# Factor w/ 4 levels "negative","somewhat_negative",..: 4 3 4 4 4 4 4 4 3 1 ...
postResample(c5ClassesRCSam,testingRCSam$galaxysentiment)
# Accuracy     Kappa 
# 0.8478822 0.6064975 

# plot takes the output of the train object and creates a plot
plot(c5_rcSam)
confusionMatrix(data = c5ClassesRCSam, testingRCSam$galaxysentiment)

# Confusion Matrix and Statistics

# Reference
# Prediction          negative somewhat_negative somewhat_positive positive
# negative               378                 1                 6       40
# somewhat_negative        2                11                 0        1
# somewhat_positive        7                 3               196       23
# positive               236               120               150     2698

# Overall Statistics

# Accuracy : 0.8479          
# 95% CI : (0.8362, 0.8591)
# No Information Rate : 0.7133          
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.6065          
# Mcnemar's Test P-Value : < 2.2e-16       

# Statistics by Class:

# Class: negative Class: somewhat_negative Class: somewhat_positive Class: positive
# Sensitivity                  0.60674                 0.081481                  0.55682          0.9768
# Specificity                  0.98553                 0.999197                  0.99062          0.5441
# Pos Pred Value               0.88941                 0.785714                  0.85590          0.8421
# Neg Pred Value               0.92892                 0.967859                  0.95718          0.9042
# Prevalence                   0.16090                 0.034866                  0.09091          0.7133
# Detection Rate               0.09762                 0.002841                  0.05062          0.6968
# Detection Prevalence         0.10976                 0.003616                  0.05914          0.8275
# Balanced Accuracy            0.79614                 0.540339                  0.77372          0.7605

c5impRCSam = varImp(c5_rcSam, scale = FALSE)
print(c5impRCSam)

# C5.0 variable importance

# only 20 most important variables shown (out of 58)

# Overall
# iphone         100.00
# googleandroid   98.81
# sonyxperia      94.99
# htcdisneg       92.45
# htcdisunc       91.91
# samsungcampos   91.45
# samsunggalaxy   91.13
# htccampos       90.74
# iphonedisunc    89.06
# iphonedispos    88.45
# htcdispos       88.11
# iphoneperpos    31.51
# iphonedisneg    28.39
# iphoneperneg    23.91
# iphonecampos    23.62
# iphonecamunc    21.79
# samsungcamneg   12.42
# samsungcamunc    8.77
# samsungperpos    7.72
# ios              7.47

plot(c5impRCSam)

#--- samsungPCA, Principal Component Analysis PCA---#
c5_pcaSam <-  train(galaxysentiment ~ .,
                data = train.pcaSam,
                method = "C5.0",
                tuneLength = 2,
                trControl = fitControl,
                # na.action = na.pass)
                na.action = na.omit)

c5_pcaSam
# C5.0 

c5ClassespcaSam <- predict(c5_pcaSam, newdata = test.pcaSam)
str(c5ClassespcaSam)
# Factor w/ 6 levels "very_negative",..: 6 1 6 4 6 6 6 6 6 6 ...
postResample(c5ClassespcaSam,test.pcaSam$galaxysentiment)
# Accuracy     Kappa 
# 0.7481271 0.4906724
# plot takes the output of the train object and creates a plot
plot(c5_pcaSam)
confusionMatrix(data = c5ClassespcaSam, test.pcaSam$galaxysentiment)

# Confusion Matrix and Statistics

# Reference
# Prediction          very_negative negative somewhat_negative somewhat_positive positive very_positive
# very_negative               327        2                 0                 4        6            35
# negative                      0        0                 0                 0        0             0
# somewhat_negative             1        0                15                 1        0             1
# somewhat_positive             6        2                 0               199        7            33
# positive                      7        3                 2                 2      104            17
# very_positive               167      107               118               146      308          2251

# Overall Statistics

# Accuracy : 0.7481          
# 95% CI : (0.7341, 0.7617)
# No Information Rate : 0.6037          
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.4907          
# Mcnemar's Test P-Value : NA 
# Statistics by Class:
  
#   Class: very_negative Class: negative Class: somewhat_negative Class: somewhat_positive Class: positive
# Sensitivity                       0.64370         0.00000                 0.111111                  0.56534         0.24471
# Specificity                       0.98602         1.00000                 0.999197                  0.98636         0.99100
# Pos Pred Value                    0.87433             NaN                 0.833333                  0.80567         0.77037
# Neg Pred Value                    0.94824         0.97055                 0.968855                  0.95778         0.91408
# Prevalence                        0.13123         0.02945                 0.034875                  0.09093         0.10979
# Detection Rate                    0.08447         0.00000                 0.003875                  0.05141         0.02687
# Detection Prevalence              0.09662         0.00000                 0.004650                  0.06381         0.03487
# Balanced Accuracy                 0.81486         0.50000                 0.555154                  0.77585         0.61785
# Class: very_positive
# Sensitivity                        0.9632
# Specificity                        0.4485
# Pos Pred Value                     0.7268
# Neg Pred Value                     0.8889
# Prevalence                         0.6037
# Detection Rate                     0.5815
# Detection Prevalence               0.8001
# Balanced Accuracy                  0.7059

c5imppcaSam = varImp(c5_pcaSam, scale = FALSE)
print(c5imppcaSam)

# C5.0 variable importance

# only 20 most important variables shown (out of 25)

# Overall
# PC6    98.99
# PC4    16.43
# PC12   14.76
# PC7    13.66
# PC22   12.39
# PC14   12.32
# PC17   11.36
# PC9     8.02
# PC2     6.00
# PC13    5.42
# PC24    3.31
# PC5     3.01
# PC11    2.82
# PC1     2.63
# PC10    1.70
# PC25    1.40
# PC18    1.34
# PC3     1.06
# PC21    1.06
# PC15    0.33

plot(c5imppcaSam)

#--- samsung, principal component analysis and recoded sentiment target---#
c5_pcarcSam <-  train(galaxysentiment ~ .,
                data = train.pcaRCSam,
                method = "C5.0",
                tuneLength = 2,
                trControl = fitControl,
                # na.action = na.pass)
                na.action = na.omit)

c5_pcarcSam

# C5.0 

# 9039 samples
# 24 predictor
# 4 classes: 'negative', 'somewhat_negative', 'somewhat_positive', 'positive' 

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8134, 8137, 8136, 8136, 8135, 8135, ... 
# Resampling results across tuning parameters:
  
#   model  winnow  trials  Accuracy   Kappa    
# rules  FALSE    1      0.8345636  0.5648140
# rules  FALSE   10      0.8354039  0.5675305
# rules   TRUE    1      0.8345747  0.5647554
# rules   TRUE   10      0.8352598  0.5676654
# tree   FALSE    1      0.8344309  0.5652810
# tree   FALSE   10      0.8350388  0.5679853
# tree    TRUE    1      0.8342536  0.5647209
# tree    TRUE   10      0.8348621  0.5672873

# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 10, model = rules and winnow = FALSE.

c5ClassespcarcSam <- predict(c5_pcarcSam, newdata = test.pcaRCSam)
str(c5ClassespcarcSam)
# Factor w/ 4 levels "negative","somewhat_negative",..: 4 3 4 4 4 4 4 4 3 1 ...

postResample(c5ClassespcarcSam,test.pcaRCSam$galaxysentiment)
# Accuracy     Kappa 
# 0.8429752 0.5913389 

# plot takes the output of the train object and creates a plot
plot(c5_pcarcSam)
confusionMatrix(data = c5ClassespcarcSam, test.pcaRCSam$galaxysentiment)

# Confusion Matrix and Statistics

# Reference
# Prediction          negative somewhat_negative somewhat_positive positive
# negative               372                 4                 2       41
# somewhat_negative        2                 9                 0        1
# somewhat_positive        3                 0               191       28
# positive               246               122               159     2692

# Overall Statistics

# Accuracy : 0.843           
# 95% CI : (0.8311, 0.8543)
# No Information Rate : 0.7133  
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.5913          
# Mcnemar's Test P-Value : NA              

# Statistics by Class:

# Class: negative Class: somewhat_negative Class: somewhat_positive Class: positive
# Sensitivity                  0.59711                 0.066667                  0.54261          0.9747
# Specificity                  0.98553                 0.999197                  0.99119          0.5252
# Pos Pred Value               0.88783                 0.750000                  0.86036          0.8363
# Neg Pred Value               0.92731                 0.967358                  0.95589          0.8928
# Prevalence                   0.16090                 0.034866                  0.09091          0.7133
# Detection Rate               0.09607                 0.002324                  0.04933          0.6952
# Detection Prevalence         0.10821                 0.003099                  0.05733          0.8314
# Balanced Accuracy            0.79132                 0.532932                  0.76690          0.7499

c5imppcarcSam = varImp(c5_pcarcSam, scale = FALSE)
print(c5imppcarcSam)
# C5.0 variable importance

# only 20 most important variables shown (out of 24)

# Overall
# PC1   100.00
# PC4   100.00
# PC6   100.00
# PC3    99.94
# PC12   99.70
# PC19   99.26
# PC15   98.12
# PC18   97.85
# PC22   94.87
# PC20   87.12
# PC16   83.26
# PC13   82.31
# PC23   28.69
# PC2    21.66
# PC21   18.43
# PC10   18.07
# PC14   17.08
# PC5    14.75
# PC9    12.83
# PC24   12.72

plot(c5imppcarcSam)

#--- Random Forest   ---#
modelLookup('rf')
#--- samsungDF, no features removed ---#

rfAllSam<-train(galaxysentiment~.,data=trainingALLsam,method="rf",
             trControl= fitControl,
             prox=TRUE,allowParallel=TRUE)

rfAllSam

# Random Forest 

# 9040 samples
# 58 predictor
# 6 classes: 'very_negative', 'negative', 'somewhat_negative', 'somewhat_positive', 'positive', 'very_positive' 

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8136, 8138, 8136, 8135, 8134, 8139, ... 
# Resampling results across tuning parameters:
  
#   mtry  Accuracy   Kappa    
# 2    0.7091173  0.3672327
# 30    0.7672465  0.5387132
# 58    0.7583408  0.5259712

# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was mtry = 30.

rfClassesSam <- predict(rfAllSam, newdata = testingALLsam)
str(rfClassesSam)
# Factor w/ 6 levels "very_negative",..: 6 1 6 4 6 6 6 6 6 6 ...

postResample(rfClassesSam, testingALLsam$galaxysentiment)
# Accuracy     Kappa 
# 0.7576854 0.5118042 

# plot(rf_Fit) takes the output of the train object and creates a plot
plot(rfAllSam)
confusionMatrix(data = rfClassesSam, testingALLsam$galaxysentiment)

# Confusion Matrix and Statistics

# Reference
# Prediction          very_negative negative somewhat_negative somewhat_positive positive very_positive
# very_negative               334        2                 0                 0        4            24
# negative                      0        0                 0                 0        0             1
# somewhat_negative             2        1                15                 1        1             3
# somewhat_positive             5        2                 0               207        5            27
# positive                      6        3                 1                 2      118            23
# very_positive               161      106               119               142      297          2259

# Overall Statistics

# Accuracy : 0.7577          
# 95% CI : (0.7439, 0.7711)
# No Information Rate : 0.6037          
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.5118          
# Mcnemar's Test P-Value : < 2.2e-16       

# Statistics by Class:

# Class: very_negative Class: negative Class: somewhat_negative Class: somewhat_positive
# Sensitivity                       0.65748       0.0000000                 0.111111                  0.58807
# Specificity                       0.99108       0.9997338                 0.997859                  0.98892
# Pos Pred Value                    0.91758       0.0000000                 0.652174                  0.84146
# Neg Pred Value                    0.95038       0.9705426                 0.968815                  0.96000
# Prevalence                        0.13123       0.0294498                 0.034875                  0.09093
# Detection Rate                    0.08628       0.0000000                 0.003875                  0.05347
# Detection Prevalence              0.09403       0.0002583                 0.005942                  0.06355
# Balanced Accuracy                 0.82428       0.4998669                 0.554485                  0.78849
# Class: positive Class: very_positive
# Sensitivity                  0.27765               0.9666
# Specificity                  0.98984               0.4622
# Pos Pred Value               0.77124               0.7325
# Neg Pred Value               0.91743               0.9009
# Prevalence                   0.10979               0.6037
# Detection Rate               0.03048               0.5836
# Detection Prevalence         0.03952               0.7967
# Balanced Accuracy            0.63375               0.7144

rfimpSam = varImp(rfAllSam, scale = FALSE)
plot(rfimpSam)

#--- samsungDF, correlated  features removed ---#

rfCORSam<-train(galaxysentiment~.,data=trainingCORsam,method="rf",
                trControl= fitControl,
                prox=TRUE,allowParallel=TRUE)

rfCORSam

# Random Forest 

# 9040 samples
# 44 predictor
# 6 classes: 'very_negative', 'negative', 'somewhat_negative', 'somewhat_positive', 'positive', 'very_positive' 

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8137, 8136, 8135, 8136, 8136, 8136, ... 
# Resampling results across tuning parameters:
  
#   mtry  Accuracy   Kappa    
# 2    0.6990501  0.3348144
# 23    0.7606422  0.5220979
# 44    0.7524350  0.5118769

# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was mtry = 23.
rfClassesCORSam <- predict(rfCORSam, newdata = testingCORsam)
str(rfClassesCORSam)
# Factor w/ 6 levels "very_negative",..: 6 4 6 6 4 1 6 6 4 6 ...

postResample(rfClassesCORSam, testingCORsam$galaxysentiment)
# Accuracy     Kappa 
# 0.7685353 0.5449627 

# plot(rf_Fit) takes the output of the train object and creates a plot
plot(rfCORSam)
confusionMatrix(data = rfClassesCORSam, testingCORsam$galaxysentiment)
# Confusion Matrix and Statistics

# Reference
# Prediction          very_negative negative somewhat_negative somewhat_positive positive very_positive
# very_negative               358        1                 1                 2        5            36
# negative                      0        0                 1                 0        0            12
# somewhat_negative             0        0                22                 0        0             5
# somewhat_positive             4        0                 1               223       11            26
# positive                      7        0                 1                 4      138            24
# very_positive               139      113               109               123      271          2234

# Overall Statistics

# Accuracy : 0.7685          
# 95% CI : (0.7549, 0.7817)
# No Information Rate : 0.6037          
# P-Value [Acc > NIR] : < 2.2e-16 
# Kappa : 0.545           
# Mcnemar's Test P-Value : NA              

# Statistics by Class:

# Class: very_negative Class: negative Class: somewhat_negative Class: somewhat_positive
# Sensitivity                       0.70472        0.000000                 0.162963                  0.63352
# Specificity                       0.98662        0.996540                 0.998662                  0.98806
# Pos Pred Value                    0.88834        0.000000                 0.814815                  0.84151
# Neg Pred Value                    0.95675        0.970451                 0.970604                  0.96423
# Prevalence                        0.13123        0.029450                 0.034875                  0.09093
# Detection Rate                    0.09248        0.000000                 0.005683                  0.05761
# Detection Prevalence              0.10411        0.003358                 0.006975                  0.06846
# Balanced Accuracy                 0.84567        0.498270                 0.580812                  0.81079
# Class: positive Class: very_positive

# Sensitivity                  0.32471               0.9559
# Specificity                  0.98955               0.5078
# Pos Pred Value               0.79310               0.7474
# Neg Pred Value               0.92237               0.8832
# Prevalence                   0.10979               0.6037
# Detection Rate               0.03565               0.5771
# Detection Prevalence         0.04495               0.7722
# Balanced Accuracy            0.65713               0.7319


rfimpCORSam = varImp(rfCORSam, scale = FALSE)
plot(rfimpCORSam)

#--- samsungDF, RFE ---#

rfRFESam<-train(galaxysentiment~.,data=trainingRFEsam,method="rf",
                trControl= fitControl,
                prox=TRUE,allowParallel=TRUE)

rfRFESam
# Random Forest 

# 9040 samples
# 11 predictor
# 6 classes: 'very_negative', 'negative', 'somewhat_negative', 'somewhat_positive', 'positive', 'very_positive' 

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8136, 8136, 8136, 8136, 8135, 8137, ... 
# Resampling results across tuning parameters:
  
#   mtry  Accuracy   Kappa

# 2    0.7376682  0.4509082
# 6    0.7511762  0.5060569
# 11    0.7455225  0.4997568

# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was mtry = 6.


rfClassesRFESam <- predict(rfRFESam, newdata = testingRFEsam)
str(rfClassesRFESam)
# Factor w/ 6 levels "very_negative",..: 6 1 6 4 4 6 6 4 1 6 ...
postResample(rfClassesRFESam,testingRFEsam$galaxysentiment)
# Accuracy     Kappa 
# 0.7597520 0.5279151 

# plot takes the output of the train object and creates a plot
plot(rfRFESam)
confusionMatrix(data = rfClassesRFESam, testingRFEsam$galaxysentiment)

# Confusion Matrix and Statistics

# Reference
# Prediction          very_negative negative somewhat_negative somewhat_positive positive very_positive
# very_negative               360        0                 1                 3        9            36
# negative                      1        0                 0                 0        0             2
# somewhat_negative             1        0                16                 0        0             3
# somewhat_positive             1        2                 4               208        8            63
# positive                      4        1                 2                 3      141            17
# very_positive               141      111               112               138      267          2216

# Overall Statistics

# Accuracy : 0.7598         
# 95% CI : (0.746, 0.7731)
# No Information Rate : 0.6037         
# P-Value [Acc > NIR] : < 2.2e-16      

# Kappa : 0.5279         
# Mcnemar's Test P-Value : NA             

# Statistics by Class:
#                     Class: very_negative Class: negative Class: somewhat_negative Class: somewhat_positive
# Sensitivity                        0.7087        0.000000                 0.118519                  0.59091
# Specificity                        0.9854        0.999201                 0.998929                  0.97783
# Pos Pred Value                     0.8802        0.000000                 0.800000                  0.72727
# Neg Pred Value                     0.9573        0.970527                 0.969099                  0.95983
# Prevalence                         0.1312        0.029450                 0.034875                  0.09093
# Detection Rate                     0.0930        0.000000                 0.004133                  0.05373
# Detection Prevalence               0.1057        0.000775                 0.005167                  0.07388
# Balanced Accuracy                  0.8470        0.499601                 0.558724                  0.78437
# Class: positive Class: very_positive
# Sensitivity                  0.33176               0.9482
# Specificity                  0.99216               0.4987
# Pos Pred Value               0.83929               0.7424
# Neg Pred Value               0.92331               0.8634
# Prevalence                   0.10979               0.6037
# Detection Rate               0.03642               0.5725
# Detection Prevalence         0.04340               0.7711
# Balanced Accuracy            0.66196               0.7235

rfimpRFESam = varImp(rfRFESam, scale = FALSE)

print(rfimpRFESam)
# rf variable importance

# Overall
# iphone         662.79
# iphonedisunc   248.54
# htcphone       228.32
# samsunggalaxy  215.95
# iphoneperpos   195.09
# googleandroid  164.72
# iphonedispos   155.77
# iphonecamneg   121.60
# iphonecampos    99.79
# htccampos       50.23
# sonyxperia      18.96
# plot(rfimpRFESam)

#--- samsungDF, NZV ---#

rfNZVSam<-train(galaxysentiment~.,data=trainingNZVsam,method="rf",
                trControl= fitControl,
                prox=TRUE,allowParallel=TRUE)

rfNZVSam

# Random Forest 

# 9040 samples
# 11 predictor
# 6 classes: 'very_negative', 'negative', 'somewhat_negative', 'somewhat_positive', 'positive', 'very_positive' 

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8135, 8138, 8137, 8137, 8134, 8136, ... 
# Resampling results across tuning parameters:
  
#   mtry  Accuracy   Kappa    
# 2    0.7551667  0.5006113
# 6    0.7516708  0.5010271
# 11    0.7451888  0.4927421

# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was mtry = 2.

rfClassesNZVSam <- predict(rfNZVSam, newdata = testingNZVsam)
str(rfClassesNZVSam)
# Factor w/ 6 levels "very_negative",..: 4 6 6 6 4 4 6 4 6 6 ...
postResample(rfClassesNZVSam,testingNZVsam$galaxysentiment)
# Accuracy     Kappa 
# 0.7556187 0.5000188 

# plot takes the output of the train object and creates a plot
plot(rfNZVSam)
confusionMatrix(data = rfClassesNZVSam, testingNZVsam$galaxysentiment)

# Confusion Matrix and Statistics

# Reference
# Prediction          very_negative negative somewhat_negative somewhat_positive positive very_positive
# very_negative               350        0                15                 0        9            23
# negative                      0        0                 0                 0        0             0
# somewhat_negative             0        0                 0                 0        0             0
# somewhat_positive             1        1                 0               164        5            17
# positive                      3        2                 0                 1      129            15
# very_positive               154      111               120               187      282          2282

# Overall Statistics

# Accuracy : 0.7556          
# 95% CI : (0.7418, 0.7691)
# No Information Rate : 0.6037          
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.5             
# Mcnemar's Test P-Value : NA   
# Statistics by Class:
  
#   Class: very_negative Class: negative Class: somewhat_negative Class: somewhat_positive
# Sensitivity                       0.68898         0.00000                  0.00000                  0.46591
# Specificity                       0.98602         1.00000                  1.00000                  0.99318
# Pos Pred Value                    0.88161             NaN                      NaN                  0.87234
# Neg Pred Value                    0.95452         0.97055                  0.96513                  0.94895
# Prevalence                        0.13123         0.02945                  0.03487                  0.09093
# Detection Rate                    0.09042         0.00000                  0.00000                  0.04237
# Detection Prevalence              0.10256         0.00000                  0.00000                  0.04857
# Balanced Accuracy                 0.83750         0.50000                  0.50000                  0.72954
# Class: positive Class: very_positive
# Sensitivity                  0.30353               0.9765
# Specificity                  0.99391               0.4433
# Pos Pred Value               0.86000               0.7277
# Neg Pred Value               0.92045               0.9252
# Prevalence                   0.10979               0.6037
# Detection Rate               0.03332               0.5895
# Detection Prevalence         0.03875               0.8101
# Balanced Accuracy            0.64872               0.7099

rfimpNZVSam = varImp(rfNZVSam, scale = FALSE)
print(rfimpNZVSam)

# rf variable importance

# Overall
# iphone         489.20
# htcphone       241.90
# samsunggalaxy  189.52
# iphonedisunc   144.95
# iphonedisneg   138.86
# iphonedispos   104.06
# iphonecamunc    96.19
# iphoneperpos    92.57
# iphonecampos    81.09
# iphoneperneg    59.93
# iphoneperunc    52.46
# plot(rfimpNZVSam)

#--- samsungDF, and recoded sentiment target  ---#

rfRCSam<-train(galaxysentiment~.,data=trainingRCSam,method="rf",
                trControl= fitControl,
                prox=TRUE,allowParallel=TRUE)

rfRCSam

# Random Forest 

# 9039 samples
# 58 predictor
# 4 classes: 'negative', 'somewhat_negative', 'somewhat_positive', 'positive' 

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8136, 8134, 8136, 8135, 8135, 8134, ... 
# Resampling results across tuning parameters:
  
#   mtry  Accuracy   Kappa    
# 2    0.7844676  0.3638956
# 30    0.8403593  0.5862950
# 58    0.8340643  0.5738207

# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was mtry = 30.

rfClassesRCSam <- predict(rfRCSam, newdata = testingRCSam)
str(rfClassesRCSam)
#  Factor w/ 4 levels "negative","somewhat_negative",..: 4 3 4 4 4 4 4 4 3 1 ...
postResample(rfClassesRCSam,testingRCSam$galaxysentiment)
# Accuracy     Kappa 
# 0.8502066 0.6136712 

# plot(rf_Fit) takes the output of the train object and creates a plot
plot(rfRCSam)
confusionMatrix(data = rfClassesRCSam, testingRCSam$galaxysentiment)

# Confusion Matrix and Statistics

# Reference
# Prediction          negative somewhat_negative somewhat_positive positive
# negative               371                 0                 3       36
# somewhat_negative        3                11                 0        6
# somewhat_positive        5                 2               213       23
# positive               244               122               136     2697

# Overall Statistics

# Accuracy : 0.8502          
# 95% CI : (0.8386, 0.8613)
# No Information Rate : 0.7133          
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.6137          
# Mcnemar's Test P-Value : < 2.2e-16       

# Statistics by Class:

# Class: negative Class: somewhat_negative Class: somewhat_positive Class: positive
# Sensitivity                  0.59551                 0.081481                  0.60511          0.9765
# Specificity                  0.98800                 0.997592                  0.99148          0.5477
# Pos Pred Value               0.90488                 0.550000                  0.87654          0.8431
# Neg Pred Value               0.92721                 0.967809                  0.96170          0.9034
# Prevalence                   0.16090                 0.034866                  0.09091          0.7133
# Detection Rate               0.09582                 0.002841                  0.05501          0.6965
# Detection Prevalence         0.10589                 0.005165                  0.06276          0.8262
# Balanced Accuracy            0.79175                 0.539537                  0.79830          0.7621

rfimpRCSam = varImp(rfRCSam, scale = FALSE)
print(rfimpRCSam)
# rf variable importance

# only 20 most important variables shown (out of 58)

# Overall
# iphone         617.23
# samsunggalaxy  235.69
# htcphone       205.22
# googleandroid  156.11
# iphoneperpos   133.85
# iphonedispos    90.73
# iphonedisunc    85.37
# iphoneperneg    63.83
# iphonecampos    56.94
# iphoneperunc    56.53
# htccampos       35.48
# iphonecamneg    31.06
# iphonecamunc    30.45
# htcdispos       28.87
# sonyxperia      26.60
# sonyperpos      14.87
# ios             12.64
# htcperpos       12.04
# samsungperpos   10.61
# plot(rfimpRCSam)

#--- samsungDF, principal component analysis PCA ---#

rfPCASam<-train(galaxysentiment~.,data=train.pcaSam,method="rf",
                trControl= fitControl,
                prox=TRUE,allowParallel=TRUE)

rfPCASam
# Random Forest 

# 9040 samples
# 25 predictor
# 6 classes: 'very_negative', 'negative', 'somewhat_negative', 'somewhat_positive', 'positive', 'very_positive' 

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8139, 8136, 8137, 8136, 8134, 8135, ... 
# Resampling results across tuning parameters:
  
#   mtry  Accuracy   Kappa    
# 2    0.7572664  0.5195446
# 13    0.7572885  0.5201348
# 25    0.7558508  0.5176772

# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was mtry = 13.

rfClassesPCASam <- predict(rfPCASam, newdata = test.pcaSam)
str(rfClassesPCASam)
# Factor w/ 6 levels "very_negative",..: 6 1 6 4 6 6 2 6 6 6 ...

postResample(rfClassesPCASam, test.pcaSam$galaxysentiment)
# Accuracy     Kappa 
# 0.7445105 0.4895649 

# plot takes the output of the train object and creates a plot
plot(rfPCASam)
confusionMatrix(data = rfClassesPCASam, test.pcaSam$galaxysentiment)

# Confusion Matrix and Statistics

# Reference
# Prediction          very_negative negative somewhat_negative somewhat_positive positive very_positive
# very_negative               335        3                 1                 1        8            49
# negative                      0        1                 0                 1        0             8
# somewhat_negative             2        0                15                 1        1             3
# somewhat_positive             2        2                 0               202        5            33
# positive                     11        2                 1                 2      108            23
# very_positive               158      106               118               145      303          2221

# Overall Statistics

# Accuracy : 0.7445          
# 95% CI : (0.7305, 0.7582)
# No Information Rate : 0.6037          
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.4896          
# Mcnemar's Test P-Value : NA              

# Statistics by Class:

#                      Class: very_negative Class: negative Class: somewhat_negative Class: somewhat_positive Class: positive
# Sensitivity                       0.65945       0.0087719                 0.111111                  0.57386         0.25412
# Specificity                       0.98156       0.9976045                 0.998126                  0.98806         0.98868
# Pos Pred Value                    0.84383       0.1000000                 0.681818                  0.82787         0.73469
# Neg Pred Value                    0.95020       0.9707330                 0.968823                  0.95864         0.91488
# Prevalence                        0.13123       0.0294498                 0.034875                  0.09093         0.10979
# Detection Rate                    0.08654       0.0002583                 0.003875                  0.05218         0.02790
# Detection Prevalence              0.10256       0.0025833                 0.005683                  0.06303         0.03797
# Balanced Accuracy                 0.82051       0.5031882                 0.554619                  0.78096         0.62140
# Class: very_positive
# Sensitivity                        0.9504
# Specificity                        0.4589
# Pos Pred Value                     0.7280
# Neg Pred Value                     0.8585
# Prevalence                         0.6037
# Detection Rate                     0.5738
# Detection Prevalence               0.7882
# Balanced Accuracy                  0.7046

rfimpPCASam = varImp(rfPCASam, scale = FALSE)
print(rfimpPCASam)
# rf variable importance

# only 20 most important variables shown (out of 25)

# Overall
# PC4   421.85
# PC3   266.47
# PC13  150.68
# PC25  130.59
# PC14  128.25
# PC2   119.86
# PC5   119.75
# PC17  118.55
# PC1   105.76
# PC7    97.84
# PC12   95.60
# PC6    92.02
# PC24   88.67
# PC9    70.77
# PC20   67.99
# PC18   67.45
# PC15   66.99
# PC8    66.58
# PC23   66.44
# PC21   64.03
plot(rfimpPCASam)

#--- samsungDF, principal component analysis and recoded sentiment target ---#

rfPCARCSam<-train(galaxysentiment~.,data=train.pcaRCSam,method="rf",
                trControl= fitControl,
                prox=TRUE,allowParallel=TRUE)

rfPCARCSam
# Random Forest 

# 9039 samples
# 24 predictor
# 4 classes: 'negative', 'somewhat_negative', 'somewhat_positive', 'positive' 

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8134, 8136, 8134, 8137, 8134, 8135, ... 
# Resampling results across tuning parameters:
  
#   mtry  Accuracy   Kappa    
# 2    0.8346838  0.5713209
# 13    0.8355911  0.5732451
# 24    0.8344402  0.5705402

# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was mtry = 13.

rfClassesPCARCSam <- predict(rfPCARCSam, newdata = test.pcaRCSam)
str(rfClassesPCARCSam)
# Factor w/ 4 levels "negative","somewhat_negative",..: 4 3 4 4 4 4 4 4 3 1 ...
postResample(rfClassesPCARCSam,test.pcaRCSam$galaxysentiment)
# Accuracy     Kappa 
# 0.8442665 0.5986580 

# plot takes the output of the train object and creates a plot
plot(rfPCARCSam)
confusionMatrix(data = rfClassesPCARCSam, test.pcaRCSam$galaxysentiment)

# Confusion Matrix and Statistics

# Reference
# Prediction          negative somewhat_negative somewhat_positive positive
# negative               370                 1                 3       43
# somewhat_negative        5                12                 1        9
# somewhat_positive        4                 0               202       25
# positive               244               122               146     2685

# Overall Statistics

# Accuracy : 0.8443          
# 95% CI : (0.8325, 0.8556)
# No Information Rate : 0.7133          
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.5987          
# Mcnemar's Test P-Value : < 2.2e-16       

# Statistics by Class:

#                      Class: negative Class: somewhat_negative Class: somewhat_positive Class: positive
# Sensitivity                  0.59390                 0.088889                  0.57386          0.9721
# Specificity                  0.98553                 0.995986                  0.99176          0.5387
# Pos Pred Value               0.88729                 0.444444                  0.87446          0.8398
# Neg Pred Value               0.92677                 0.968010                  0.95880          0.8859
# Prevalence                   0.16090                 0.034866                  0.09091          0.7133
# Detection Rate               0.09556                 0.003099                  0.05217          0.6934
# Detection Prevalence         0.10770                 0.006973                  0.05966          0.8257
# Balanced Accuracy            0.78972                 0.542437                  0.78281          0.7554


rfimpARCSam = varImp(rfPCARCSam, scale = FALSE)
print(rfimpARCSam)

# rf variable importance

# only 20 most important variables shown (out of 24)

# Overall
# PC4   486.05
# PC3   239.46
# PC10  126.73
# PC12  123.35
# PC15   97.60
# PC6    93.76
# PC23   79.20
# PC8    75.25
# PC18   69.81
# PC16   69.79
# PC17   66.27
# PC13   64.56
# PC1    60.39
# PC19   59.59
# PC24   57.53
# PC5    55.66
# PC2    50.63
# PC20   49.45
# PC14   49.33
# PC21   46.66

plot(rfimpARCSam)

#--- k-Nearest Neighbors   ---#
modelLookup("kknn")

#--- samsung, no features removed ---#

knn_ALLsam <- train(galaxysentiment ~., data = trainingALLsam, method = "kknn",
                  trControl=fitControl,
                  na.action = na.omit,
                  #  kmax = 9)
                  preProcess = c("center", "scale"))
# tuneLength = 10)

knn_ALLsam

# k-Nearest Neighbors 

# 9040 samples
# 58 predictor
# 6 classes: 'very_negative', 'negative', 'somewhat_negative', 'somewhat_positive', 'positive', 'very_positive' 

# Pre-processing: centered (58), scaled (58) 
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8138, 8136, 8135, 8136, 8136, 8134, ... 
# Resampling results across tuning parameters:
  
#   kmax  Accuracy   Kappa    
# 5     0.6058724  0.3725475
# 7     0.6400560  0.4068162
# 9     0.6974022  0.4624917

# Tuning parameter 'distance' was held constant at a value of 2
# Tuning parameter 'kernel' was held constant at
# a value of optimal
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were kmax = 9, distance = 2 and kernel = optimal.  

knnClassesSam <- predict(knn_ALLsam, newdata = testingALLsam)
str(knnClassesSam)
# Factor w/ 6 levels "very_negative",..: 6 1 6 4 6 6 6 6 6 6 ...
postResample(knnClassesSam,testingALLsam$galaxysentiment)
# Accuracy     Kappa 
# 0.7403772 0.4880874

# plot takes the output of the train object and creates a plot
plot(knn_ALLsam)
confusionMatrix(data = knnClassesSam, testingALLsam$galaxysentiment)
# Confusion Matrix and Statistics

# Reference
# Prediction          very_negative negative somewhat_negative somewhat_positive positive very_positive
# very_negative               338        5                 1                 0       16            42
# negative                      0        0                 0                 0        2             5
# somewhat_negative             1        1                15                 3        2            10
# somewhat_positive             5        1                 3               208        5            52
# positive                      9        2                 2                 1      109            32
# very_positive               155      105               114               140      291          2196

# Overall Statistics

# Accuracy : 0.7404          
# 95% CI : (0.7263, 0.7541)
# No Information Rate : 0.6037          
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.4881          
# Mcnemar's Test P-Value : < 2.2e-16       

# Statistics by Class:

# Class: very_negative Class: negative Class: somewhat_negative Class: somewhat_positive
# Sensitivity                       0.66535        0.000000                 0.111111                  0.59091
# Specificity                       0.98097        0.998137                 0.995450                  0.98124
# Pos Pred Value                    0.84080        0.000000                 0.468750                  0.75912
# Neg Pred Value                    0.95099        0.970497                 0.968742                  0.95997
# Prevalence                        0.13123        0.029450                 0.034875                  0.09093
# Detection Rate                    0.08732        0.000000                 0.003875                  0.05373
# Detection Prevalence              0.10385        0.001808                 0.008267                  0.07078
# Balanced Accuracy                 0.82316        0.499068                 0.553280                  0.78608

#                   Class: positive Class: very_positive
# Sensitivity                  0.25647               0.9397
# Specificity                  0.98665               0.4752
# Pos Pred Value               0.70323               0.7318
# Neg Pred Value               0.91496               0.8379
# Prevalence                   0.10979               0.6037
# Detection Rate               0.02816               0.5673
# Detection Prevalence         0.04004               0.7753
# Balanced Accuracy            0.62156               0.7074

knnimpSam = varImp(knn_ALLsam, scale = FALSE)
print(knnimpSam)

# ROC curve variable importance

# variables are sorted by maximum importance across the classes
# only 20 most important variables shown (out of 58)

# very_negative negative somewhat_negative somewhat_positive positive very_positive
# htcphone             0.7253   0.7253            0.7253            0.7253   0.7253        0.7232
# iphone               0.6797   0.6815            0.7212            0.6797   0.6797        0.6815
# iphonedisunc         0.5038   0.5749            0.6639            0.5077   0.5068        0.5749
# samsunggalaxy        0.6583   0.6583            0.6583            0.6583   0.6583        0.6526
# iphonedisneg         0.5055   0.5111            0.6540            0.5055   0.5055        0.5111
# iphonedispos         0.5028   0.5764            0.6406            0.5032   0.5017        0.5764
# iphonecamunc         0.5373   0.5339            0.6269            0.5390   0.5339        0.5373
# iphonecamneg         0.5313   0.5345            0.6154            0.5314   0.5313        0.5345
# htcdispos            0.6063   0.6063            0.6063            0.6063   0.6063        0.6040
# htcperpos            0.6062   0.6062            0.6062            0.6062   0.6062        0.6046
# htccampos            0.6011   0.6008            0.6008            0.6022   0.6008        0.6011
# iphonecampos         0.5499   0.5422            0.5972            0.5396   0.5380        0.5499
# htcperneg            0.5943   0.5927            0.5927            0.5927   0.5927        0.5943
# htcdisneg            0.5874   0.5852            0.5852            0.5866   0.5852        0.5874
# htcdisunc            0.5815   0.5812            0.5812            0.5816   0.5812        0.5815
# htccamneg            0.5804   0.5786            0.5786            0.5790   0.5786        0.5804
# htcperunc            0.5791   0.5790            0.5790            0.5792   0.5790        0.5791
# htccamunc            0.5746   0.5741            0.5741            0.5749   0.5741        0.5746
# iphoneperpos         0.5187   0.5234            0.5696            0.5286   0.5174        0.5234
# sonyxperia           0.5676   0.5676            0.5676            0.5676   0.5676        0.5663
    
plot(knnimpSam)  


#--- Support Vector Machine   ---#
modelLookup("svmLinear2")

#--- samsung, no features removed ---#

svm_ALLsam <- train(galaxysentiment ~., data = trainingALLsam, method = "svmLinear2",
                 trControl=fitControl,
                 na.action = na.omit,
                 preProcess = c("center", "scale"))




svm_ALLsam
# Support Vector Machines with Linear Kernel 

# 9040 samples
# 58 predictor
# 6 classes: 'very_negative', 'negative', 'somewhat_negative', 'somewhat_positive', 'positive', 'very_positive' 

# Pre-processing: centered (58), scaled (58) 
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8137, 8138, 8136, 8135, 8136, 8137, ... 
# Resampling results across tuning parameters:
  
#   cost  Accuracy   Kappa    
# 0.25  0.7132509  0.3906825
# 0.50  0.7165697  0.4026543
# 1.00  0.7120902  0.3968908

# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was cost = 0.5.

svmClassesSam <- predict(svm_ALLsam, newdata = testingALLsam)
str(svmClassesSam)
# Factor w/ 6 levels "very_negative",..: 6 1 6 6 6 6 6 6 6 6 ...

postResample(svmClassesSam,testingALLsam$galaxysentiment)
# Accuracy     Kappa 
# 0.7101524 0.3869717

# plot(rf_Fit) takes the output of the train object and creates a plot
plot(svm_ALLsam)
confusionMatrix(data = svmClassesSam,testingALLsam$galaxysentiment)
# Confusion Matrix and Statistics

# Reference
# Prediction          very_negative negative somewhat_negative somewhat_positive positive very_positive
# very_negative               326        4                 2                 6       14            55
# negative                      0        0                 1                 0        0             0
# somewhat_negative             0        1                 1                 0        1             0
# somewhat_positive             4        2                13                95        2            11
# positive                      5        0                 0                 1       62             6
# very_positive               173      107               118               250      346          2265

# Overall Statistics

# Accuracy : 0.7102          
# 95% CI : (0.6956, 0.7244)
# No Information Rate : 0.6037          
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.387           
# Mcnemar's Test P-Value : NA              

# Statistics by Class:
#                      Class: very_negative Class: negative Class: somewhat_negative Class: somewhat_positive
# Sensitivity                       0.64173       0.0000000                0.0074074                  0.26989
# Specificity                       0.97591       0.9997338                0.9994647                  0.99091
# Pos Pred Value                    0.80098       0.0000000                0.3333333                  0.74803
# Neg Pred Value                    0.94746       0.9705426                0.9653568                  0.93136
# Prevalence                        0.13123       0.0294498                0.0348747                  0.09093
# Detection Rate                    0.08422       0.0000000                0.0002583                  0.02454
# Detection Prevalence              0.10514       0.0002583                0.0007750                  0.03281
# Balanced Accuracy                 0.80882       0.4998669                0.5034360                  0.63040
# Class: positive Class: very_positive
# Sensitivity                  0.14588               0.9692
# Specificity                  0.99652               0.3520
# Pos Pred Value               0.83784               0.6950
# Neg Pred Value               0.90440               0.8824
# Prevalence                   0.10979               0.6037
# Detection Rate               0.01602               0.5851
# Detection Prevalence         0.01912               0.8419
# Balanced Accuracy            0.57120               0.6606

svmimpSam = varImp(svm_ALLsam, scale = FALSE)
print(svmimpSam)
# ROC curve variable importance

# variables are sorted by maximum importance across the classes
# only 20 most important variables shown (out of 58)

# very_negative negative somewhat_negative somewhat_positive positive very_positive
# htcphone             0.7253   0.7253            0.7253            0.7253   0.7253        0.7232
# iphone               0.6797   0.6815            0.7212            0.6797   0.6797        0.6815
# iphonedisunc         0.5038   0.5749            0.6639            0.5077   0.5068        0.5749
# samsunggalaxy        0.6583   0.6583            0.6583            0.6583   0.6583        0.6526
# iphonedisneg         0.5055   0.5111            0.6540            0.5055   0.5055        0.5111
# iphonedispos         0.5028   0.5764            0.6406            0.5032   0.5017        0.5764
# iphonecamunc         0.5373   0.5339            0.6269            0.5390   0.5339        0.5373
# iphonecamneg         0.5313   0.5345            0.6154            0.5314   0.5313        0.5345
# htcdispos            0.6063   0.6063            0.6063            0.6063   0.6063        0.6040
# htcperpos            0.6062   0.6062            0.6062            0.6062   0.6062        0.6046
# htccampos            0.6011   0.6008            0.6008            0.6022   0.6008        0.6011
# iphonecampos         0.5499   0.5422            0.5972            0.5396   0.5380        0.5499
# htcperneg            0.5943   0.5927            0.5927            0.5927   0.5927        0.5943
# htcdisneg            0.5874   0.5852            0.5852            0.5866   0.5852        0.5874
# htcdisunc            0.5815   0.5812            0.5812            0.5816   0.5812        0.5815
# htccamneg            0.5804   0.5786            0.5786            0.5790   0.5786        0.5804
# htcperunc            0.5791   0.5790            0.5790            0.5792   0.5790        0.5791
# htccamunc            0.5746   0.5741            0.5741            0.5749   0.5741        0.5746
# iphoneperpos         0.5187   0.5234            0.5696            0.5286   0.5174        0.5234
# sonyxperia           0.5676   0.5676            0.5676            0.5676   0.5676        0.5663

plot(svmimpSam)

#################
# Evaluate models
#################

##--- Compare iphone models ---##

# use resamples to compare model performance, compare all  models
ModelFitiphone <- resamples(list(c5=c5_all, c5cor=c5_cor, c5nze=c5_nze, c5rfe=c5_rfe, c5rc = c5_rc, c5pca=c5_pca, c5pcarc=c5_pcarc, rf=rfAll, rfcor=rf_cor, rfnzv=rf_nzv, rfrfe=rf_rfe, rfrc=rf_rc, rfpca=rf_pca,  knn=knn_ALL, svm=svm_ALL))

# output summary metrics for tuned models 
summary(ModelFitiphone)
# Call:
#   summary.resamples(object = ModelFitiphone)

# Models: c5, c5cor, c5nze, c5rfe, c5rc, c5pca, c5pcarc, rf, rfcor, rfnzv, rfrfe, rfrc, rfpca, knn, svm 
# Number of resamples: 100


# Accuracy 
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# c5    0.7439560 0.7670253 0.7710511 0.7712433 0.7767390 0.7916207    0
# c5cor 0.7480748 0.7629548 0.7677490 0.7684332 0.7742291 0.7984581    0
# c5nze 0.7334802 0.7488987 0.7546752 0.7546858 0.7601214 0.7830396    0
# c5rfe 0.7488987 0.7622452 0.7684272 0.7681730 0.7748146 0.7918502    0
# c5rc  0.8270925 0.8437844 0.8500551 0.8492897 0.8543448 0.8689427    0
# c5pca 0.7414741 0.7531678 0.7579758 0.7592330 0.7645765 0.7825607    0
# c5pcarc 0.8215859 0.8359031 0.8419608 0.8413747 0.8467897 0.8656388    0
# rf    0.7442117 0.7642522 0.7694003 0.7709129 0.7768302 0.7935982    0
# rfcor 0.7442117 0.7634112 0.7706274 0.7702197 0.7788779 0.7909791    0
# rfnzv 0.7370166 0.7504121 0.7568757 0.7562810 0.7629774 0.7744774    0
# rfrfe 0.7414741 0.7617738 0.7674930 0.7677505 0.7737148 0.7923077    0
# rfrc  0.8303965 0.8457299 0.8514034 0.8513715 0.8577729 0.8744493    0
# rfpca 0.7378855 0.7532360 0.7603078 0.7597721 0.7658874 0.7819383    0
# knn   0.3406593 0.3597360 0.3654376 0.3677853 0.3763412 0.3997797    0
# svm   0.6791621 0.7051705 0.7105117 0.7109988 0.7161716 0.7356828    0

# Kappa
#            Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# c5    0.4942551 0.5449443 0.5531423 0.5549480 0.5684419 0.6014158    0
# c5cor 0.5010965 0.5371936 0.5504260 0.5495705 0.5634886 0.6136916    0
# c5nze 0.4725847 0.5040041 0.5187784 0.5177828 0.5294733 0.5829375    0
# c5rfe 0.5000857 0.5329800 0.5483841 0.5479132 0.5633372 0.6018209    0
# c5rc  0.5565068 0.6102768 0.6264678 0.6250556 0.6404291 0.6779249    0
# c5pca 0.4840379 0.5194034 0.5296977 0.5320031 0.5443726 0.5841904    0
# c5pcarc 0.5433772 0.5870456 0.6030126 0.6030573 0.6168638 0.6693224    0
# rf    0.5041473 0.5428314 0.5575498 0.5581979 0.5712665 0.6088173    0
# rfcor 0.4934278 0.5412238 0.5598753 0.5571688 0.5755811 0.6015126    0
# rfnzv 0.4733676 0.5060969 0.5207543 0.5187863 0.5334464 0.5601668    0
# rfrfe 0.4926986 0.5393138 0.5519768 0.5523206 0.5637755 0.6053954    0
# rfrc  0.5689346 0.6132768 0.6332152 0.6321344 0.6499264 0.6994006    0
# rfpca 0.4920829 0.5234393 0.5396780 0.5383356 0.5522868 0.5893918    0
# knn   0.1459564 0.1714659 0.1826882 0.1823216 0.1914063 0.2165592    0
# svm   0.3418768 0.4026862 0.4158367 0.4160263 0.4302014 0.4813787    0


##--- Compare samsung galaxy models ---##

# use resamples to compare model performance, compare all  models
ModelFitSamsung <- resamples(list(c5Sam=c5_allSam, c5corSam=c5_corSam, c5nzeSam=c5_nzvSam, c5rfeSam=c5_rfeSam, c5rcSam = c5_rcSam, c5pcaSam=c5_pcaSam,   c5pcarcSam=c5_pcarcSam,  rfSam=rfAllSam, rfcorSam=rfCORSam, rfnzvSam=rfNZVSam, rfrfeSam=rfRFESam,  rfrcSam=rfRCSam, rfpcaSam=rfPCASam, rfpcarcSam=rfPCARCSam, knnSam=knn_ALLsam, svmSam=svm_ALLsam))
#     
# output summary metrics for tuned models 
summary(ModelFitSamsung)

# Call:
#   summary.resamples(object = ModelFitSamsung)

# Models: c5Sam, c5corSam, c5nzeSam, c5rfeSam, c5rcSam, c5pcaSam, c5pcarcSam, rfSam, rfcorSam, rfnzvSam, rfrfeSam, rfrcSam, rfpcaSam, rfpcarcSam, knnSam, svmSam 
# Number of resamples: 100 

# Accuracy 
#                Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# c5Sam    0.7447514 0.7615621 0.7686773 0.7687487 0.7758433 0.7911602    0
# c5corSam 0.7414365 0.7588496 0.7656163 0.7644465 0.7705056 0.7884828    0
# c5nzeSam 0.7323009 0.7461326 0.7520752 0.7525334 0.7578771 0.7732301    0
# c5rfeSam 0.7289823 0.7438384 0.7526228 0.7510284 0.7577434 0.7723757    0
# c5rcSam  0.8121547 0.8340708 0.8405316 0.8398280 0.8464732 0.8637874    0
# c5pcaSam 0.7364341 0.7530454 0.7592697 0.7595899 0.7677633 0.7809735    0
# c5pcarcSam 0.8112583 0.8296460 0.8349945 0.8354039 0.8405757 0.8571429    0
# rfSam    0.7477876 0.7612600 0.7665929 0.7672465 0.7732301 0.7942478    0
# rfcorSam 0.7356195 0.7544248 0.7603765 0.7606422 0.7666574 0.7953540    0
# rfnzvSam 0.7339956 0.7477178 0.7553956 0.7551667 0.7622338 0.7787611    0
# rfrfeSam   0.7242525 0.7436464 0.7520752 0.7511762 0.7589162 0.7782705    0
# rfrcSam  0.8165746 0.8340708 0.8403548 0.8403593 0.8462389 0.8629834    0
# rfpcaSam 0.7303867 0.7507609 0.7569061 0.7572885 0.7644456 0.7834254    0
# rfpcarcSam 0.8176796 0.8279781 0.8351770 0.8355911 0.8431104 0.8604651    0
# knnSam   0.4243094 0.7313430 0.7435052 0.6974022 0.7539356 0.7776549    0
# svmSam   0.6795580 0.7112832 0.7179204 0.7165697 0.7232216 0.7422566    0


# Kappa 
#               Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# c5Sam    0.4845835 0.5206424 0.5397134 0.5379672 0.5532299 0.5850507    0
# c5corSam 0.4716159 0.5099573 0.5288115 0.5255351 0.5409586 0.5812339    0
# c5nzeSam 0.4537164 0.4825284 0.4986838 0.4985842 0.5125857 0.5446331    0
# c5rfeSam 0.4436890 0.4858618 0.5056916 0.5018850 0.5184853 0.5581505    0
# c5rcSam  0.4963746 0.5635673 0.5836745 0.5816485 0.6003488 0.6516832    0
# c5pcaSam 0.4609347 0.5031781 0.5150767 0.5175607 0.5352369 0.5670939    0
# c5pcarcSam 0.4949136 0.5528178 0.5659446 0.5675305 0.5847619 0.6297451    0
# rfSam    0.4907483 0.5238933 0.5380496 0.5387132 0.5526199 0.5974922    0
# rfcorSam 0.4566351 0.5085095 0.5222664 0.5220979 0.5362633 0.6016663    0
# rfnzvSam 0.4505349 0.4825234 0.5000795 0.5006113 0.5179015 0.5589781    0
# rfrfeSam   0.4372832 0.4914665 0.5066746 0.5060569 0.5223944 0.5640690    0
# rfrcSam  0.5153949 0.5690855 0.5871946 0.5862950 0.6046678 0.6556634    0
# rfpcaSam 0.4618452 0.5053673 0.5179852 0.5201348 0.5349838 0.5783554    0
# rfpcarcSam 0.5211362 0.5508882 0.5716917 0.5732451 0.5963948 0.6516950    0
# knnSam   0.1872572 0.4752174 0.5017349 0.4624917 0.5201503 0.5759672    0
# svmSam   0.3218206 0.3893116 0.4030034 0.4026543 0.4179521 0.4561894    0



##--- Conclusion iphone sentiment---##
# The top model is random forest with the target variable altered wtih recoded sentiment.  

##--- Save top performing model iphone ---##

# save model 
saveRDS(ModelFitiphone, "ModelFitResultsiphone.rds")
ModelFitResultsiphone <- readRDS("ModelFitResultsiphone.rds")


##--- Conclusion samsung sentiment---##
# The top model is C5.0 with the target variable altered wtih recoded sentiment.  

##--- Save top performing model samsung ---##

# save model 
saveRDS(ModelFitSamsung, "ModelFitResultsSamsung.rds")
ModelFitResultsSamsung <- readRDS("ModelFitResultsSamsung.rds")

#################
# Test Set
#################
# See individual executions for results



##############################################
# Predict with iPhone sentiment with top model
# Predict with test set/ validation
##############################################

# make predictions with iphoneLargeMatrix data
iphonePred1 <- predict(rf_rc, newdata = iphoneLargeMatrix)
str(iphonePred1)

plot(iphonePred1,iphoneLargeMatrix$iphonesentiment)
plot(iphonePred1)
# print predictions
print(iphonePred1)

# negative somewhat_negative somewhat_positive          positive 
# 14638              1279               310              7326 

summary(iphonePred1)

pieDataIPhone<- data.frame(COM = c("negative", "somewhat negative", "somewhat positive", "positive"), 
                            values = c(14638, 1279, 310, 7326 ))

# create pie chart iPhone
plot_ly(pieDataIPhone, labels = ~COM, values = ~ values, type = "pie",
        textposition = 'inside',
        textinfo = 'label+percent',
        insidetextfont = list(color = '#FFFFFF'),
        hoverinfo = 'text',
        text = ~paste( values),
        marker = list(colors = colors,
                      line = list(color = '#FFFFFF', width = 1)),
        showlegend = F) %>%
  layout(title = 'iPhone Sentiment', 
         xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
         yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))




##############################################
# Predict with Galaxy sentiment with top model
# Predict with test set/ validation
##############################################

# make predictions with galaxyLargeMatrix data
galaxyPred1 <- predict(c5_rcSam, newdata = galaxyLargeMatrix)
str(galaxyPred1)

plot(galaxyPred1,galaxyLargeMatrix$galaxysentiment)
plot(galaxyPred1)


# print predictions
print(galaxyPred1)

  
summary(galaxyPred1)
# negative somewhat_negative   somewhat_positive          positive 
# 14413              1290               312              7538     

pieDataGalaxy <- data.frame(COM = c("negative", "somewhat negative", "somewhat positive", "positive"), 
                            values = c(14413, 1290, 312, 7593 ))

# create pie chart galaxy data
plot_ly(pieDataGalaxy, labels = ~COM, values = ~ values, type = "pie",
              textposition = 'inside',
              textinfo = 'label+percent',
              insidetextfont = list(color = '#FFFFFF'),
              hoverinfo = 'text',
              text = ~paste( values),
              marker = list(colors = colors,
                            line = list(color = '#FFFFFF', width = 1)),
              showlegend = F) %>%
  layout(title = 'Galaxy Sentiment', 
         xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
         yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))




