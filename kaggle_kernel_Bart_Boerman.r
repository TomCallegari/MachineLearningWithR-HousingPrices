#
#
#
#
#
# Kaggle Kernel - data.table & caretEnsemble (House Prices - Ames, Iowa) - Bart Boerman

require(knitr)              ## dynamic report generation in R
require(DT)                 ## display data in html tables
require(ggplot2)            ## plotting 
require(gridExtra)          ## arrange visualizations using grid 
require(dplyr)              ## easy data wrangling on small data frames
require(data.table)         ## fast data wrangling and analysis
require(funModeling)        ## table with counts on missing values. Do not load before psych or caret. It will mask some stuff.
require(psych)              ## descriptive statistics, skewness and kurtosis
require(caret)              ## (near) zero variance, train and predict
require(caretEnsemble)      ## ensemble modelling
require(xgboost)
require(glmnet)
require(LiblineaR)          ## svm
require(vtreat)             ## one hot encode and ignore factor levels with low frequency

setwd("D://Analytics/Housing Prices")

train.dt <- fread(input = "train.csv", 
                  sep = ",", 
                  nrows = -1,
                  header = T,
                  na.strings=c("NA","N/A","null"),
                  stringsAsFactors = F,
                  check.names = T,
                  strip.white = T,
                  blank.lines.skip = T,
                  data.table = T) 

test.dt <- fread(input = "test.csv", 
                 sep = ",", 
                 nrows = -1,
                 header = T,
                 na.strings=c("NA","N/A","null"),
                 stringsAsFactors = F,
                 check.names = T,
                 strip.white = T,
                 blank.lines.skip = T,
                 data.table = T)

# Create one data set for feature engineering. 
train.dt[, dataPartition:="train"]
test.dt[, SalePrice:=as.integer(NA)] 
test.dt[, dataPartition:="test"]
full.dt <- rbindlist(list(train.dt, test.dt), use.names = F, fill = F)

# Square Footage
variablesSquareFootage <- c(
  "LotFrontage", 		## Linear feet of street connected to property 
  "LotArea",    		## Lot size in square feet
  "MasVnrArea",  		## Masonry veneer area in square feet
  "BsmtFinSF1",		  ## Type 1 finished square feet	
  "BsmtFinSF2",		  ## Type 2 finished square feet
  "BsmtUnfSF",		  ## Unfinished square feet of basement area
  "TotalBsmtSF", 		## Total square feet of basement area
  "FirstFlrSF",		  ## First Floor square feet
  "SecondFlrSF",	  ## Second floor square feet
  "LowQualFinSF", 	## Low quality finished square feet (all floors)
  "GrLivArea", 		  ## Above grade (ground) living area square feet
  "GarageArea",     ## Size of garage in square feet
  "WoodDeckSF",     ## Wood deck area in square feet
  "OpenPorchSF",    ## Open porch area in square feet  
  "EnclosedPorch",  ## Enclosed porch area in square feet 
  "ThreeSsnPorch",  ## Three season porch area in square feet 
  "ScreenPorch",    ## Screen porch area in square feet
  "PoolArea" 		    ## Pool area in square feet
)

# Count variables
variablesCounts <- c(
  "BsmtFullBath",		## Basement full bathrooms
  "BsmtHalfBath",		## Basement half bathrooms
  "FullBath",			  ## Full bathrooms above grade
  "HalfBath",			  ## Half baths above grade
  "BedroomAbvGr",		## Bedrooms above grade (does NOT include basement bedrooms)
  "KitchenAbvGr",		## Kitchens above grade
  "TotRmsAbvGrd",		## Total rooms above grade (does not include bathrooms)
  "Fireplaces",		  ## Number of fireplaces
  "GarageCars"     	## Size of garage in car capacity
)

# Value variables
variablesValues <- c(
  "MiscVal",        ## $ Value of miscellaneous feature
  "SalePrice"       ## $ Price paid
)

# Categorical and Ordinal variables
variablesFactor <- colnames(full.dt)[which(as.vector(full.dt[,sapply(full.dt, class)]) == "character")]
variablesFactor <- setdiff(variablesFactor, "dataPartition") 
variablesFactor <- c(variablesFactor,
                     ## variables with data type integer which are factors
                     "MSSubClass",     ## Identifies the type of dwelling involved in the sale
                     "OverallQual",    ## Rates the overall material and finish of the house
                     "OverallCond"     ## Rates the overall condition of the house
)

# R does not support variable names starting with a number
setnames(full.dt, c("X1stFlrSF","X2ndFlrSF","X3SsnPorch"), c("FirstFlrSF","SecondFlrSF","ThreeSsnPorch"))

# Data cleaning typos
full.dt[YearRemodAdd > YrSold, YearRemodAdd:= YrSold] ## Fix typo
full.dt[GarageYrBlt == 2207, GarageYrBlt:= 2007] ## Fix typo
full.dt[MSSubClass  == 150, MSSubClass:= 160] ## 150 not in training set
full.dt[Exterior1st  == "Wd Sdng", Exterior1st:= "WdSdng"] ## Fix spaces
full.dt[Exterior2nd  == "Wd Sdng", Exterior2nd:= "WdSdng"] ## Fix spaces
full.dt[Exterior2nd  == "Brk Cmn", Exterior2nd:= "BrkComm"] ## Fix typo
full.dt[Exterior2nd  == "Wd Shng", Exterior2nd:= "WdShing"] ## Fix typo
full.dt[RoofMatl  == "Tar&Grv", RoofMatl:= "TarGrv"] ## Fix '&'
full.dt[RoofMatl  == "WdShngl", RoofMatl:= "WdShing"] ## See exterior

## (Ordinal) factors 

# Since categorical variables enter into statistical models differently than continuous variables, 
# storing data as factors insures that the modeling functions will treat such data correctly. 

# The code performs the following tasks: rename variable names, change data type to factor and order ordinal factors. 


changeColType <- variablesFactor
full.dt[,(changeColType):= lapply(.SD, as.factor), .SDcols = changeColType]
## Set columns to numeric
changeColType <- c(variablesSquareFootage, variablesCounts, variablesValues)
full.dt[,(changeColType):= lapply(.SD, as.numeric), .SDcols = changeColType]


# An ordered factor is used to represent an ordinal variable.


## OverallQual, rates the overall material and finish of the house
full.dt[,OverallQual:=ordered(OverallQual, levels = c(1:10))]
## OverallCond, rates the overall condition of the house
full.dt[,OverallCond:=ordered(OverallCond, levels = c(1:10))]
## KitchenQual, kitchen quality
full.dt[,KitchenQual:=ordered(KitchenQual, levels = c("Po","Fa","TA","Gd","Ex"))]
## GarageFinish (contains NA's)
full.dt[,GarageFinish:=ordered(GarageFinish, levels = c("None","Unf","RFn","Fin"))]
## GarageQual
full.dt[,GarageQual:=ordered(GarageQual, levels = c("None","Po","Fa","TA","Gd","Ex"))]
## GarageCond
full.dt[,GarageCond:=ordered(GarageCond, levels = c("None","Po","Fa","TA","Gd","Ex"))]
## ExterQual, evaluates the quality of the material on the exterior  
full.dt[,ExterQual:=ordered(ExterQual, levels = c("Po","Fa","TA","Gd","Ex"))]
## ExterCond, evaluates the present condition of the material on the exterior
full.dt[,ExterCond:=ordered(ExterCond, levels = c("Po","Fa","TA","Gd","Ex"))]
## BsmtQual (contains NA's), evaluates the height of the basement
full.dt[,BsmtQual:=ordered(BsmtQual, levels = c("None","Po","Fa","TA","Gd","Ex"))]
## BsmtCond (contains NA's), evaluates the general condition of the basement
full.dt[,BsmtCond:=ordered(BsmtCond, levels = c("None","Po","Fa","TA","Gd","Ex"))]
## BsmtExposure (contains NA's), refers to walkout or garden level walls
full.dt[,BsmtExposure:=ordered(BsmtExposure, levels = c("None","No","Mn","Av","Gd"))]
## BsmtFinType1 (contains NA's), rating of basement finished area
full.dt[,BsmtFinType1:=ordered(BsmtFinType1, levels = c("None","Unf","LwQ","Rec","BLQ","ALQ","GLQ"))]
## FireplaceQu (contains NA's), fireplace quality
full.dt[,FireplaceQu:=ordered(FireplaceQu, levels = c("None","Po","Fa","TA","Gd","Ex"))]
## Electrical
full.dt[,Electrical:=ordered(Electrical, levels = c("FuseP","Mix","FuseF","FuseA","SBrkr"))]
## Fence
full.dt[,Fence:=ordered(Fence, levels = c("None","MnWw","MnPrv","GdWo","GdPrv"))]
## PoolQC
full.dt[,PoolQC:=ordered(PoolQC, levels = c("None","Fa","Gd","Ex"))]


# Descriptive table
descStats <- describe(full.dt[, c(variablesSquareFootage,variablesValues), with = FALSE]) 
datatable(round(descStats,2), rownames = T,
          caption = "Descriptive statistics", 
          options = list(pageLength = 8)) ## Interactive HTML table

## (Near) zero variance

# Variables with zero variance are mostly constant across the data set, hence might provide little information 
# and potentially cause overfitting. The table is generated on the training data with help of the caret package.


zeroVarianceVariables.df <- nearZeroVar(train.dt, names = T, saveMetrics = T,
                                        foreach = T, allowParallel = T)
datatable(round(subset(zeroVarianceVariables.df, nzv == TRUE, 
                       select =     c("freqRatio","percentUnique")),2), 
          rownames = T,
          caption = "Variables with (near) zero variance", 
          options = list(pageLength = 8))

## Impute with default, median, mode or regression


#### function to find the mode.
findMode <- function(x) {
  names(table(x))[table(x)==max(table(x))]
}

#### imputations

## Kitchen
full.dt[is.na(KitchenQual), KitchenQual := findMode(full.dt$KitchenQual) ] ## One record, set to Typical

## Garage
full.dt[is.na(GarageFinish) & GarageType == "Detchd", ':=' (GarageFinish = "Fin",
                                                            GarageCars = 1,
                                                            GarageArea = 360,
                                                            GarageYrBlt = YearBuilt,
                                                            GarageQual = findMode(full.dt$GarageQual),
                                                            GarageCond = findMode(full.dt$GarageCond))] 
full.dt[is.na(GarageFinish), GarageFinish := "None"]
full.dt[is.na(GarageQual), GarageQual := "None"]
full.dt[is.na(GarageCond), GarageCond := "None"]
full.dt[is.na(GarageType), GarageType := "None"]
full.dt[is.na(GarageYrBlt), GarageYrBlt := 0]

## Basement
full.dt[is.na(BsmtExposure) & BsmtFinType1 == "Unf" , BsmtExposure := "No"]
full.dt[is.na(BsmtExposure), BsmtExposure := "None"]
full.dt[is.na(BsmtQual) & BsmtFinType1 == "Unf" , BsmtQual := "TA"]
full.dt[is.na(BsmtQual), BsmtQual := "None"]
full.dt[is.na(BsmtCond), BsmtCond := "None"]
full.dt[is.na(BsmtFinType1), BsmtFinType1 := "None"]
full.dt[is.na(BsmtFinType2) & BsmtFinSF2 > 0, BsmtFinType2 := "Unf"]
full.dt[is.na(BsmtFinType2), BsmtFinType2 := "None"]
full.dt[is.na(BsmtFinSF1),':=' (BsmtFinSF1 = 0, BsmtFinSF2 = 0, BsmtUnfSF = 0, TotalBsmtSF = 0)] 
full.dt[is.na(BsmtFullBath),':=' (BsmtFullBath = 0, BsmtHalfBath = 0)] 

## FireplaceQu  
full.dt[is.na(FireplaceQu), FireplaceQu := "None"]

## MSZoning
## RL for missing MSZoning in Mitchel because GrLivArea is greater then max of RM
## Not sure (yet) for missing MSZoning in IDOTRR. RM is most common in IDOTRR but might be wrong
full.dt[is.na(MSZoning) & Neighborhood == "Mitchel", MSZoning := "RL"]
full.dt[is.na(MSZoning) & Neighborhood == "IDOTRR", MSZoning  := "RM"]

## Electrical
## Most common value for neighborhood Timber is SBrkr
full.dt[is.na(Electrical) , Electrical  := findMode(full.dt$Electrical)]

## Exterior
## Most common for neighborhood and large total square footage is "MetalSd"
full.dt[is.na(Exterior1st),':=' (Exterior1st = findMode(full.dt$Exterior1st),Exterior2nd = findMode(full.dt$Exterior2nd))]
## MasVnrType and MasVnrArea. Taking the easy way out here
full.dt[is.na(MasVnrType),':=' (MasVnrType = "None", MasVnrArea = 0)]
## SaleType
full.dt[is.na(SaleType), SaleType := findMode(full.dt$SaleType)]
## Functional
full.dt[is.na(Functional), Functional := findMode(full.dt$Functional)]
## MiscFeature
full.dt[is.na(MiscFeature), MiscFeature := "None"]
## Alley
full.dt[is.na(Alley), Alley := "None"]
## Utilities
full.dt[is.na(Utilities), Utilities := findMode(full.dt$Utilities)]
## PoolQC
full.dt[is.na(PoolQC), PoolQC := "None"]
## Fence
full.dt[is.na(Fence), Fence := "None"]
## LotFrontage
## Alternative 1, impute by the median per neigborhood 
# full.dt[, LotFrontage := replace(LotFrontage, is.na(LotFrontage), median(LotFrontage, na.rm=TRUE)), by=.(Neighborhood)]
## Alternatove 2, impute with logistic regression
fit <- lm(log1p(LotFrontage) ~ log1p(LotArea) + LotConfig, data = full.dt[!is.na(LotFrontage),])
full.dt[is.na(LotFrontage), LotFrontage :=  round(expm1(predict(fit, newdata = full.dt[is.na(LotFrontage),])),0 )]

# Correlation {.tabset}

# High multicollinearity potentially increase the variance of the coefficient estimates and makes the estimates 
# sensitive to minor changes in the model. The result is that the model becomes unstable. It does not affect the 
# fit of the model or the quality of the predictions. 

## correlation matrix

require(corrplot)
## calculate correlation matrix
variablesNumeric <- sapply(full.dt, is.numeric) ## both numeric integers
corrData <- full.dt[dataPartition == "train", ..variablesNumeric]
corMatrix = cor(corrData)

## plot correlation
corrplot(corMatrix,
         title = "",
         type = "lower", 
         order = "hclust", 
         hclust.method = "centroid",
         tl.cex = 0.8,
         tl.col = "black", 
         tl.srt = 45)

## select highly correlated values

## table with highly correlated values
corTable <- setDT(melt(corMatrix))[order(-value)][value!=1] 
corTableSalePrice <- corTable[Var1=="SalePrice",][order(-value)]
(corTableHigh <- corTable[value > 0.7 | value < -0.70 ][order(Var1,-value)])

## manually selection highly correlated variables to be removed during pre-processing
variablesHighlyCorrelated <- c(
  "TotalBsmtSF",    ## high correlation	with FirstFlrSF
  "GarageArea",     ## high correlation GarageCars
  "TotRmsAbvGrd" ,  ## high correlation GrLivArea
  "GarageCond",     ## high correlation with GarageQual
  "GarageYrBlt",    ## high correlation with GarageQual
  "YearRemodAdd",   ## correlated with qualities and interaction variable age 
  "BsmtFinSF1"      ## correlated with 	BsmtFinType1
)

# Feature engineering
houseStyle.bin <- c("1Story" = "1Story", 
                    "1.5Fin" = "1.5Story", 
                    "1.5Unf" = "1.5Story",
                    "2.5Unf" = "2.5Story",
                    "2.5Fin" = "2.5Story",
                    "2Story" = "2Story",
                    "SFoyer" = "SFoyer",
                    "SLvl" = "SLvl") 
full.dt[, ':=' (
  age = YrSold - YearRemodAdd,
  isRemodeled = ifelse(YearRemodAdd == YearBuilt, 1, 0),
  isNew       = ifelse(YrSold       == YearBuilt, 1, 0),
  overallQualGood    = ifelse(as.integer(OverallQual) - 5 < 0, 0, as.integer(OverallQual) - 5),
  overallQualBad     = ifelse(5 - as.integer(OverallQual) < 0, 0, 5 - as.integer(OverallQual)),
  # sfPorch = EnclosedPorch + ThreeSsnPorch + ScreenPorch,
  sfTotal     = (TotalBsmtSF + FirstFlrSF + SecondFlrSF),  
  hasUnfinishedLevel = ifelse(HouseStyle %in% c("1.5Unf","2.5Unf"),1,0),
  HouseStyle = as.factor(houseStyle.bin[HouseStyle]),
  countBathrooms = FullBath + HalfBath + BsmtHalfBath + BsmtFullBath,
  averageRoomSizeAbvGrd = GrLivArea / TotRmsAbvGrd,
  bathRoomToRoomAbvGrd = (FullBath + HalfBath) / TotRmsAbvGrd,
  landscapeInteraction = as.integer(LotShape) * as.integer(LandContour),
  garageInteraction = GarageCars * as.integer(GarageQual),
  yrMoSoldInt = as.numeric(format(as.Date(paste(YrSold, MoSold, "1", sep="-")), '%Y%m')) ,
  #### Added or changed 
  sfPorch = EnclosedPorch + ThreeSsnPorch + ScreenPorch + OpenPorchSF,
  isNewerDwelling = ifelse(MSSubClass %in% c(20,60,120),1,0),
  isRemodeledRecent = ifelse(YearRemodAdd == YrSold, 1, 0),
  isConditionNearRail    = ifelse(Condition1 %in% c("RRAe","RRAn","RRNe","RRNn") | Condition2 %in% c("RRAe","RRAn","RRNe","RRNn"),1,0),
  isConditionNearArtery  = ifelse(Condition1 == "Artery" | Condition2 == "Artery",1,0),
  isConditionNearFeedr  = ifelse(Condition1 == "Feedr" | Condition2 == "Feedr",1,0),
  isConditionNearPosFeature  = ifelse(Condition1 %in% c("PosA"," PosN") | Condition2 %in% c("PosA"," PosN"),1,0),
  soldHighSeason = ifelse(MoSold %in% c(5,6,7),1,0),
  yearsSinceRemodeled = ifelse(YearRemodAdd == YearBuilt, YrSold - YearRemodAdd, 0),
  scoreExterior = as.integer(ExterQual) * as.integer(ExterCond)
)]

# Data preparation {.tabset}

## Features and response

#### select features
variablesDrop <- NA
response <- "SalePrice"  
features <- setdiff(names(full.dt), c(response,variablesDrop, variablesHighlyCorrelated, "Id","dataPartition"))
#### create index for to split full data set into train and test
setkey(full.dt,dataPartition)
#### backup to be used for residual analysis
full.backup.dt <- copy(full.dt)

## Outliers

##### remove outliers
#require(broom)
## method one: highest cooks distance of residuals
#formula.all <- as.formula(paste("SalePrice ~ ", paste(features, collapse= "+")))
#modelLm <- lm(formula.all, data = full.dt["train"])

#outliersHigh.Id <- full.dt["train"]
#outliersHigh.Id$cooksd <- augment(modelLm)[[".cooksd"]]
#outliersHigh.Id <- outliersHigh.Id %>% arrange(desc(cooksd)) %>% head(10) %>% select(Id)
## method two: manual selection based on visual inspection of the data
#outliersHigh.Id <-  train.dt[GrLivArea > 4000 | LotArea > 100000 | X1stFlrSF > 3000 | GarageArea > 1200]
outliersHigh.Id <-  train.dt[GrLivArea > 4000,Id ]
full.dt <- full.dt[!(Id %in% outliersHigh.Id),]

## Ordinal factors

# Converting ordinal factors to integers gives a small performance boost when modelling glm and xgboost algorithms.

changeColType <- setDT(data.frame(sapply(full.dt,is.ordered)), keep.rownames = TRUE)[sapply.full.dt..is.ordered.==TRUE]$rn
full.dt[,(changeColType):= lapply(.SD, as.integer), .SDcols = changeColType]


## Skewed variables

skewedVariables <- sapply(full.dt[, c(variablesSquareFootage,variablesValues), with = FALSE],function(x){skew(x,na.rm=TRUE)}) ## including response variable
## keep only features that exceed a threshold for skewness
skewedVariables <- skewedVariables[skewedVariables > 0.50]
## transform excessively skewed features with log1p
skewedVariables <- names(skewedVariables)
full.dt[, (skewedVariables) := lapply(.SD, function(x) log1p(x)), .SDcols = skewedVariables]

## Scale


#### scale
varScale <- setdiff(c(variablesSquareFootage, variablesValues), c(response)) ## Do not scale response
full.dt[, (varScale) := lapply(.SD, function(x) scale(x, center = T, scale = T)), .SDcols = varScale]

## Split data into train and test

# plit in train and test after engineering. Split by key is fasted method.
train.full.dt <- full.dt["train"]
test.dt <- full.dt["test"]

## random split training into train and validate
#set.seed(333)
#n <- nrow(train.full.dt)
#shuffled.dt <- train.full.dt[sample(n), ]
#train_indices <- 1:round(0.8 * n)
#train.dt <- shuffled.dt[train_indices, ]
#validate_indices <- (round(0.8 * n) + 1):n
#validate.dt <- shuffled.dt[validate_indices, ]

## One-hot encoding

## one hot encode and ignore factor levels with low frequency
treatplan <- designTreatmentsZ(train.full.dt, minFraction = 0.01, rareCount = 0, features, verbose = FALSE)
train.full.treat <- prepare(treatplan, dframe = train.full.dt, codeRestriction = c("clean", "lev"))
test.treat  <- prepare(treatplan, dframe = test.dt, codeRestriction = c("clean", "lev"))

# Machine learning model

# Train the model
trControl <- trainControl(
  method="cv",
  number=7,
  savePredictions="final",
  index = createResample(train.full.dt$OverallQual, 7),  
  allowParallel =TRUE
)

xgbTreeGrid <- expand.grid(nrounds = 400,
                           max_depth = seq(2, 6, by = 1),
                           eta = 0.1, 
                           gamma = 0, 
                           colsample_bytree = 1.0, 
                           subsample = 1.0, 
                           min_child_weight = 4)

glmnetGridElastic <- expand.grid(.alpha = 0.3, 
                                 .lambda = 0.009) ## notice the . before the parameter

glmnetGridLasso <- expand.grid(.alpha = 1, 
                               .lambda = seq(0.001,0.1,by = 0.001))

glmnetGridRidge <- expand.grid(.alpha = 0,
                               .lambda = seq(0.001,0.1,by = 0.001))

set.seed(333)
modelList <<- caretList(
  x = train.full.treat,
  y = train.full.dt$SalePrice,
  trControl=trControl,
  metric="RMSE",
  tuneList=list(
    xgbTree = caretModelSpec(method="xgbTree",  
                             tuneGrid = xgbTreeGrid,
                             nthread = 8),
    glmnet=caretModelSpec(method="glmnet", 
                          tuneGrid = glmnetGridElastic),
    glmnet=caretModelSpec(method="glmnet", 
                          tuneGrid = glmnetGridLasso), 
    glmnet=caretModelSpec(method="glmnet", 
                          tuneGrid = glmnetGridRidge) 
    #svmLinear3= caretModelSpec(method="svmLinear3", tuneLenght = 20) ## SVM 
  )
)

## Correlation

modelCor(resamples(modelList))


## Performance summary

summary(resamples(modelList))[[3]][2:3]


# Weighted ensemble

set.seed(333)
greedyEnsemble <- caretEnsemble(
  modelList, 
  metric="RMSE",
  trControl=trainControl(
    number=7, method = "cv"
  ))

summary(greedyEnsemble)

# Analyse residuals of ensemble {.tabset}

## Outliers in predictions

tmp <- full.backup.dt["train"]
tmp <- tmp[!(Id %in% outliersHigh.Id)]
tmp$pred <- expm1(predict(greedyEnsemble, newdata = train.full.treat))
## Residual = Observed value - Predicted value 
tmp$residual <- tmp$SalePrice - tmp$pred
residualOutliers.Id <- tmp[residual <= -60000 | residual >= 100000,Id]

ggplot(tmp, aes(x = pred, y = residual)) + 
  geom_pointrange(aes(ymin = 0, ymax = residual)) + 
  geom_hline(yintercept = 0, linetype = 3) + 
  geom_text(aes(label=ifelse(Id %in% (residualOutliers.Id), Id,"")), vjust=1.5, col = "red") +
  ggtitle("Residuals vs. model prediction") +
  xlab("prediction") +
  ylab("residual") +
  theme(text = element_text(size=9)) 

## Residuals by overall quality

ggplot(tmp, aes(x = residual, y = OverallQual)) +
  geom_point() + 
  geom_text(aes(label=ifelse(Id %in% (residualOutliers.Id),Id,"")), vjust=1.5, col = "red") +
  ggtitle("Residuals vs. overall quality") +
  xlab("residual") +
  ylab("overall quality") +
  theme(text = element_text(size=9))      


## Residuals by sale type  

ggplot(tmp, aes(sample=residual, colour = tmp$SaleType)) +
  geom_point(stat = "qq",shape=1) + 
  scale_color_manual(breaks = c(levels(tmp$SaleType)),
                     values=c("gray","gray","gray" ,"gray" ,"gray" ,"gray","blue" ,"gray","red"))  



## Residuals by sale condition

ggplot(tmp, aes(sample=residual, colour = tmp$SaleCondition)) +
  geom_point(stat = "qq",shape=1) +
  scale_color_manual(breaks = c(levels(tmp$SaleCondition)),
                     values=c("gray"  ,"gray"   ,"gray"    ,"gray"   ,"blue"  ,"red")) 


## Q-Q plot residuals

qqnorm(resid(greedyEnsemble$ens_model$finalModel), 
       ylab="standardized Residuals", 
       xlab="normal Scores", 
       main="Q-Q plot residuals") 
qqline(resid(greedyEnsemble$ens_model$finalModel))


# Feature importance {.tabset}

## XGBoost

featureImp <- varImp(greedyEnsemble$models$xgbTree)
ggplot(featureImp, mapping = NULL,
       top = dim(featureImp$importance)[1]-(dim(featureImp$importance)[1]-25), environment = NULL) +
  xlab("Feature") +
  ylab("Importace") +
  theme(text = element_text(size=9))


## Lasso regression (GLM)

featureImp <- varImp(greedyEnsemble$models$glmnet)
ggplot(featureImp, mapping = NULL,
       top = dim(featureImp$importance)[1]-(dim(featureImp$importance)[1]-25), environment = NULL) +
  xlab("Feature") +
  ylab("Importace") +
  theme(text = element_text(size=9))


## Ridge regression (GLM)

featureImp <- varImp(greedyEnsemble$models$glmnet.1)
ggplot(featureImp, mapping = NULL,
       top = dim(featureImp$importance)[1]-(dim(featureImp$importance)[1]-25), environment = NULL) +
  xlab("Feature") +
  ylab("Importace") +
  theme(text = element_text(size=9))


# Evaluation

# The dwellings in our top residuals share the following characteristics: most garages can fit more then one car, 
# where not completed when last assessed (associated with new dwellings) and most have a good quality score (7+) 
# combined with an average condition (5). It may be worth mentioning that the house style is mostly "1.5 story". 

# The biggest outliers are almost all dwellings with SaleCondition equal to "Partial". Currenty I remove these 
# ourliers before submitting. However, this is a brutal fix. 

# All three models seem to graps different parts of the data. The GLM based models foces a lot on Neighborhood. 


full.dt <- full.backup.dt ## run model again without outliers detected during residual analysis. Will build function 
                          # later on. For now just give it a try.
##### remove outliers
outliersHigh.Id <-  train.dt[GrLivArea > 4000,Id ]
full.dt <- full.dt[!(Id %in% outliersHigh.Id) & !(Id %in% residualOutliers.Id),]
changeColType <- setDT(data.frame(sapply(full.dt,is.ordered)), keep.rownames = TRUE)[sapply.full.dt..is.ordered.==TRUE]$rn
full.dt[,(changeColType):= lapply(.SD, as.integer), .SDcols = changeColType]
skewedVariables <- sapply(full.dt[, c(variablesSquareFootage,variablesValues), with = FALSE],function(x){skew(x,na.rm=TRUE)}) ## including response variable
## keep only features that exceed a threshold for skewness
skewedVariables <- skewedVariables[skewedVariables > 0.50]
## transform excessively skewed features with log1p
skewedVariables <- names(skewedVariables)
full.dt[, (skewedVariables) := lapply(.SD, function(x) log1p(x)), .SDcols = skewedVariables]
#### scale
varScale <- setdiff(c(variablesSquareFootage, variablesValues), c(response)) ## Do not scale response
full.dt[, (varScale) := lapply(.SD, function(x) scale(x, center = T, scale = T)), .SDcols = varScale]
train.full.dt <- full.dt["train"]
test.dt <- full.dt["test"]
require(vtreat)
## one hot encode and ignore factor levels with low frequency
treatplan <- designTreatmentsZ(train.full.dt, minFraction = 0.01, rareCount = 0, features, verbose = FALSE)
train.full.treat <- prepare(treatplan, dframe = train.full.dt, codeRestriction = c("clean", "lev"))
test.treat  <- prepare(treatplan, dframe = test.dt, codeRestriction = c("clean", "lev"))
trControl <- trainControl(
  method="cv",
  number=7,
  savePredictions="final",
  index=createResample(train.full.dt$OverallQual, 7),  
  allowParallel =TRUE
)
set.seed(333)
modelList <<- caretList(
  x = train.full.treat,
  y = train.full.dt$SalePrice,
  trControl=trControl,
  metric="RMSE",
  tuneList=list(
    ## Do not use custom names in list. Will give prediction error with greedy ensemble. Bug in caret.
    xgbTree = caretModelSpec(method="xgbTree",  tuneGrid = xgbTreeGrid, nthread = 8),
    glmnet=caretModelSpec(method="glmnet", tuneGrid = glmnetGridElastic), ## Elastic, highly correlated with lasso and ridge regressions
    glmnet=caretModelSpec(method="glmnet", tuneGrid = glmnetGridLasso), ## Lasso
    glmnet=caretModelSpec(method="glmnet", tuneGrid = glmnetGridRidge) ## Ridge
    #svmLinear3= caretModelSpec(method="svmLinear3", tuneLenght = 20) ## SVM 
  )
)

set.seed(333)
greedyEnsemble <- caretEnsemble(
  modelList, 
  metric="RMSE",
  trControl=trainControl(
    number=7, method = "cv"
  ))

#### submit
finalPredictions <- predict(greedyEnsemble, newdata=test.treat)
finalPredictions <- data.frame(finalPredictions)
names(finalPredictions) <- "SalePrice"
finalPredictions$SalePrice <- expm1(finalPredictions$SalePrice) 
submission <- cbind(test.dt[, "Id"],finalPredictions)

write.csv(x = submission, file = "submission.csv", row.names = F)

#### save data for later use in other kernels
write.csv(x = train.full.treat, file = "train.full.treat.csv", row.names = F)
write.csv(x = test.treat, file = "test.treat.csv", row.names = F)
write.csv(x = train.full.dt, file = "train.full.dt.csv", row.names = F)
write.csv(x = test.dt, file = "test.dt.csv", row.names = F)

