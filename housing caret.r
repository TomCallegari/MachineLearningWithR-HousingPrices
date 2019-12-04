
# Housing w/ caret

library(plyr)
library(dplyr)
library(ggplot2)
library(caret)
library(doSNOW)
library(xgboost)
library(Amelia)

setwd("D:/Analytics/Housing Prices")

train <- read.csv("train.csv")

str(train)
missmap(train)

train$Id <- NULL
train$MSSubClass <- as.numeric(train$MSSubClass)
train$LotFrontage <- as.numeric(train$LotFrontage)
train$LotArea <- as.numeric(train$LotArea)
train$OverallQual <- as.numeric(train$OverallQual)
train$OverallCond <- as.numeric(train$OverallCond)
train$YearBuilt <- as.numeric(train$YearBuilt)
train$YearRemodAdd <- as.numeric(train$YearRemodAdd)
train$MasVnrArea <- as.numeric(train$MasVnrArea)
train$BsmtFinSF1 <- as.numeric(train$BsmtFinSF1)
train$BsmtFinSF2 <- as.numeric(train$BsmtFinSF2)
train$BsmtUnfSF <- as.numeric(train$BsmtUnfSF)
train$TotalBsmtSF <- as.numeric(train$TotalBsmtSF)
train$X1stFlrSF <- as.numeric(train$X1stFlrSF)
train$X2ndFlrSF <- as.numeric(train$X2ndFlrSF)
train$LowQualFinSF <- as.numeric(train$LowQualFinSF)
train$GrLivArea <- as.numeric(train$GrLivArea)
train$BsmtFullBath <- as.numeric(train$BsmtFullBath)
train$BsmtHalfBath <- as.numeric(train$BsmtHalfBath)
train$FullBath <- as.numeric(train$FullBath)
train$HalfBath <- as.numeric(train$HalfBath)
train$BedroomAbvGr <- as.numeric(train$BedroomAbvGr)
train$KitchenAbvGr <- as.numeric(train$KitchenAbvGr)
train$TotRmsAbvGrd <- as.numeric(train$TotalBsmtSF)
train$Fireplaces <- as.numeric(train$Fireplaces)
train$GarageYrBlt <- as.numeric(train$GarageYrBlt)
train$GarageCars <- as.numeric(train$GarageCars)
train$GarageArea <- as.numeric(train$GarageArea)
train$WoodDeckSF <- as.numeric(train$WoodDeckSF)
train$OpenPorchSF <- as.numeric(train$OpenPorchSF)
train$EnclosedPorch <- as.numeric(train$EnclosedPorch)
train$X3SsnPorch <- as.numeric(train$X3SsnPorch)
train$ScreenPorch <- as.numeric(train$ScreenPorch)
train$PoolArea <- as.numeric(train$PoolArea)
train$MiscVal <- as.numeric(train$MiscVal)
train$MoSold <- as.numeric(train$MoSold)
train$YrSold <- as.numeric(train$YrSold)
train$SalePrice <- as.numeric(train$SalePrice)

train$MSZoning <- revalue(train$MSZoning, c("C (all)" = "C"))
train$RoofMatl <- revalue(train$RoofMatl, c("Tar&Grv" = "Tar"))
train$Exterior1st <- revalue(train$Exterior1st, c("Wd Sdng" = "wdSdng"))
train$Exterior2nd <- revalue(train$Exterior2nd, c("Wd Sdng" = "WdSdng"))
train$Exterior2nd <- revalue(train$Exterior2nd, c("Wd Shng" = "WdShng"))
train$Exterior2nd <- revalue(train$Exterior2nd, c("Brk Cmn" = "BrkCmn"))
train$Electrical[is.na(train$Electrical)] = "SBrkr"

train$Alley <- NULL
train$LotFrontage <- NULL
train$MasVnrArea <- NULL
train$MasVnrType <- NULL
train$BsmtFinType1 <- NULL
train$BsmtFinType2 <- NULL
train$BsmtExposure <- NULL
train$BsmtCond <- NULL
train$BsmtQual <- NULL
train$PoolQC <- NULL
train$MiscFeature <- NULL
train$Fence <- NULL
train$FireplaceQu <- NULL
train$GarageCond <- NULL
train$GarageQual <- NULL
train$GarageFinish <- NULL
train$GarageYrBlt <- NULL
train$GarageType <- NULL

nulls <- select(train, Alley, LotFrontage, MasVnrArea, MasVnrType, BsmtFinType1, BsmtFinType2,
                BsmtExposure, BsmtCond, BsmtQual, PoolQC, MiscFeature, Fence, FireplaceQu, GarageCond,
                GarageQual, GarageFinish, GarageYrBlt, GarageType)

c1 <- makeCluster(3, type = "SOCK")
registerDoSNOW(c1)

stopCluster(c1)


str(nulls)

missmap(nulls)

dummy.vars <- dummyVars(~., data = train)
train.dummy <- predict(dummy.vars, train)
train.dummy <- as.data.frame(train.dummy)

c1 <- makeCluster(4, type = "SOCK")
registerDoSNOW(c1)

pre.process <- preProcess(train.dummy, method = "bagImpute")
imputed.data <- predict(pre.process, train.dummy)

stopCluster(c1)

missmap(train.dummy)

set.seed(12345)
indexes <- createDataPartition(train.dummy$SalePrice,
                               times = 1,
                               p = 0.7,
                               list = FALSE)

house.train <- train.dummy[indexes,]
house.test <- train.dummy[-indexes,]

cv.10.fold <- createMultiFolds(house.train$SalePrice, k = 10, times = 10)

train.control <- trainControl(method = "repreatedcv",
                              number = 10,
                              repeats = 3,
                              search = "grid",
                              index = cv.10.fold)

tune.grid <- expand.grid(eta = c(0.05, 0.075, 0.1),
                         nrounds = c(50, 75, 100),
                         max_depth = 6:8,
                         min_child_weight = c(2.0, 2.25, 2.5),
                         colsample_bytree = c(0.3, 0.4, 0.5),
                         gamma = 0,
                         subsample = 1)

c1 <- makeCluster(4, type = "SOCK")
registerDoSNOW(c1)

house.xgb <- train(SalePrice ~.,
                   data = house.train,
                   method = "xgbTree",
                   tuneGrid = tune.grid,
                   trControl = train.control)

stopCluster(c1)

house.xgb

preds <- predict(house.xgb, house.test)

###### Kaggle example
######
######

train <- read.csv("train.csv", header = TRUE, stringsAsFactors = FALSE)
test <- read.csv("test.csv", header = TRUE, stringsAsFactors = FALSE)

# Combine Datasets
df.combined <- rbind(within(train, rm("Id", "SalePrice")), within(test, rm("Id"))) # For kaggle example
df.combined <- train[, 2:80] # For xgboost model

dim(df.combined)
plot(df.combined$Utilities)

# Look at NA's from each column
na.cols <- which(colSums(is.na(df.combined)) > 0)
sort(colSums(sapply(df.combined[na.cols], is.na)), decreasing = TRUE)
length(na.cols)

#
# Deal with each column's missing values (NA's)

# Function for plotting categoric data
plot.categoric <- function(cols, df){
  for (col in cols) {
    order.cols <- names(sort(table(df.combined[,col]), decreasing = TRUE))
    
    num.plot <- qplot(df[,col]) +
      geom_bar(fill = 'cornflowerblue') +
      geom_text(aes(label = ..count..), stat='count', vjust=-0.5) +
      theme_minimal() +
      scale_y_continuous(limits = c(0,max(table(df[,col]))*1.1)) +
      scale_x_discrete(limits = order.cols) +
      xlab(col) +
      theme(axis.text.x = element_text(angle = 30, size=12))
    
    print(num.plot)
  }
}

# PoolQC - Pool Quality
plot.categoric("PoolQC", df.combined)
df.combined[(df.combined$PoolArea > 0) & is.na(df.combined$PoolQC), c("PoolQC", "PoolArea")]
df.combined[,c("PoolQC", "PoolArea")] %>%
  group_by(PoolQC) %>%
  summarise(mean = mean(PoolArea), counts = n())
df.combined[2421, "PoolQC"] = "Ex"
df.combined[2504, "PoolQC"] = "Ex"
df.combined[2600, "PoolQC"] = "Fa"
df.combined$PoolQC[is.na(df.combined$PoolQC)] = "None"

# Garage
length(which(df.combined$GarageYrBlt == df.combined$YearBuilt))

idx <- which(is.na(df.combined$GarageYrBlt))
df.combined[idx, "GarageYrBlt"] <- df.combined[idx, "YearBuilt"]
df.combined$GarageyrBlt <- NULL

garage.cols <- c("GarageArea", "GarageCars", "GarageQual", "GarageFinish", "GarageCond",
                 "GarageType")
df.combined[is.na(df.combined$GarageCond), garage.cols]

idx <- which(((df.combined$GarageArea < 370) & (df.combined$GarageArea > 350)) & (df.combined$GarageCars == 1))
names(sapply(df.combined[idx, garage.cols], function(x) sort(table(x), decreasing = TRUE) [1]))

df.combined[2127, "GarageQual"] = "TA"
df.combined[2127, "GarageFinish"] = "Unf"
df.combined[2127, "GarageCond"] = "TA"

for (col in garage.cols) {
  if (sapply(df.combined[col], is.numeric) == TRUE) {
    df.combined[sapply(df.combined[col], is.na), col] = 0
  }
  else {
    df.combined[sapply(df.combined[col], is.na), col] = "None"
  }
}

# KitchenQual
plot.categoric("KitchenQual", df.combined)
df.combined$KitchenQual[is.na(df.combined$KitchenQual)] = "TA"

# Electrical
plot.categoric("Electrical", df.combined)
df.combined$Electrical[is.na(df.combined$Electrical)] = "SBrkr"

# Basement Variables
bsmt.cols <- names(df.combined)[sapply(names(df.combined), function(x) str_detect(x, "Bsmt"))]
df.combined[is.na(df.combined$BsmtExposure), bsmt.cols]

plot.categoric("BsmtExposure", df.combined)
df.combined[c(949, 1488, 2349), "BsmtExposure"] = "No"

df.combined[c(949, 1488, 2349), 'BsmtExposure'] = 'No'

for (col in bsmt.cols){
  if (sapply(df.combined[col], is.numeric) == TRUE){
    df.combined[sapply(df.combined[col], is.na),col] = 0
  }
  else{
    df.combined[sapply(df.combined[col],is.na),col] = 'None'
  }
}

# Exterior1st & Exterior2nd
idx <- which(is.na(df.combined$Exterior1st) | is.na(df.combined$Exterior2nd))
df.combined[idx, c("Exterior1st", "Exterior2nd")]

df.combined$Exterior1st[is.na(df.combined$Exterior1st)] = "Other"
df.combined$Exterior2nd[is.na(df.combined$Exterior2nd)] = "Other"

# SaleType
plot.categoric("SaleType", df.combined)
df.combined[is.na(df.combined$SaleType), c("SaleCondition")]
table(df.combined$SaleCondition, df.combined$SaleType)

df.combined$SaleType[is.na(df.combined$SaleType)] = "WD"
plot.categoric("Functional", df.combined)

df.combined$Functional[is.na(df.combined$Functional)] = "Typ"

# Utilities
plot.categoric('Utilities', df.combined)

which(df.combined$Utilities == "NoSeWa")

col.drops <- c("Utilities")
df.combined <- df.combined[, !names(df.combined) %in% c("Utilities")]

# MSZoning
df.combined[is.na(df.combined$MSZoning), c("MSZoning", "MSSubClass")]

plot.categoric("MSZoning", df.combined)

table(df.combined$MSZoning, df.combined$MSSubClass)

df.combined$MSZoning[c(2217, 2905)] = "RL"
df.combined$MSZoning[c(1916, 2251)] = "RM"

# MaxVnrType - Masonry Veneer Type
df.combined[(is.na(df.combined$MasVnrType)) | (is.na(df.combined$MasVnrArea)), 
            c("MasVnrType", "MasVnrArea")]

na.omit(df.combined[,c("MasVnrType", "MasVnrArea")]) %>%
  group_by(na.omit(MasVnrType)) %>%
  summarise(MedianArea = median(MasVnrArea, na.rm = TRUE), counts = n()) %>%
  arrange(MedianArea)

plot.categoric("MasVnrType", df.combined)

df.combined[2611, "MasVnrType"] = "BrkFace"

df.combined$MasVnrType[is.na(df.combined$MasVnrType)] = "None"
df.combined$MasVnrArea[is.na(df.combined$MasVnrArea)] = 0

# LotFrontage - Linear feet of stree connected property
df.combined["Nbrh.factor"] <- factor(df.combined$Neighborhood, 
                                     levels = unique(df.combined$Neighborhood))

lot.by.nbrh <- df.combined[, c("Neighborhood", "LotFrontage")] %>%
  group_by(Neighborhood) %>%
  summarise(median = median(LotFrontage, na.rm = TRUE))
lot.by.nbrh

idx <- which(is.na(df.combined$LotFrontage))

for (i in idx){
  lot.median <- lot.by.nbrh[lot.by.nbrh == df.combined$Neighborhood[i],'median']
  df.combined[i,'LotFrontage'] <- lot.median[[1]]
}

# Fence - Fence quality
plot.categoric("Fence", df.combined)

df.combined$Fence[is.na(df.combined$Fence)] = "None"

# Misc Features
table(df.combined$MiscFeature)

df.combined$MiscFeature[is.na(df.combined$MiscFeature)] = "None"

# Fireplaces 
plot.categoric("FireplaceQu", df.combined)

which((df.combined$Fireplaces > 0) & (is.na(df.combined$FireplaceQu)))

df.combined$FireplaceQu[is.na(df.combined$FireplaceQu)] = "None"

# Alley
plot.categoric("Alley", df.combined)

df.combined$Alley[is.na(df.combined$Alley)] = "None"

paste("There are", sum(sapply(df.combined, is.na)), "missing values left")
missmap(df.combined)

###
###
### XGBoost

dummy.vars <- dummyVars(~., data = train)
train.dummy <- predict(dummy.vars, train)
train.dummy <- as.data.frame(train.dummy)

c1 <- makeCluster(4, type = "SOCK")
registerDoSNOW(c1)

pre.process <- preProcess(train.dummy, method = "bagImpute")
imputed.data <- predict(pre.process, train.dummy)

stopCluster(c1)

missmap(train.dummy)

set.seed(12345)
indexes <- createDataPartition(train.dummy$SalePrice,
                               times = 1,
                               p = 0.7,
                               list = FALSE)

house.train <- train.dummy[indexes,]
house.test <- train.dummy[-indexes,]

cv.10.fold <- createMultiFolds(house.train$SalePrice, k = 10, times = 10)

train.control <- trainControl(method = "repreatedcv",
                              number = 10,
                              repeats = 3,
                              search = "grid",
                              index = cv.10.fold)

tune.grid <- expand.grid(eta = c(0.05, 0.075, 0.1),
                         nrounds = c(50, 75, 100),
                         max_depth = 6:8,
                         min_child_weight = c(2.0, 2.25, 2.5),
                         colsample_bytree = c(0.3, 0.4, 0.5),
                         gamma = 0,
                         subsample = 1)

c1 <- makeCluster(4, type = "SOCK")
registerDoSNOW(c1)

house.xgb <- train(SalePrice ~.,
                   data = house.train,
                   method = "xgbTree",
                   tuneGrid = tune.grid,
                   trControl = train.control)

stopCluster(c1)

house.xgb

preds <- predict(house.xgb, house.test)






