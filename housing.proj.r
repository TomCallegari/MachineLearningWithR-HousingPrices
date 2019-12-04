#
#
#
#

# Import libraries
library(psych)
library(gridExtra)
library(doSNOW)
library(caret)
library(caretEnsemble)
library(Boruta)
library(kernlab)
library(randomForest)
library(gbm)
library(randomForest)
library(tidyverse)
library(DataExplorer)

# Set working directory
setwd("D://Analytics/Housing Prices")

# Import train.csv
train_raw <- read.csv("train.csv",
                  header = T,
                  stringsAsFactors = F)

# Import test.csv
test_raw <- read.csv("test.csv",
                 header = T,
                 stringsAsFactors = F)

#---------- Create Data Partition within Train and Test  ----------# 
train <- train_raw %>%
  mutate(dataPartition = "Train")
test <- test_raw %>%
  mutate(dataPartition = "Test",
         SalePrice = "NA") # Note: A SalePrice column must be addeed to test for binding to train

# Inspect Train
introduce(train)
str(train)
plot_histogram(train)
plot_missing(train)

# Inspect Test
introduce(test)
str(test)
plot_histogram(train)
plot_missing(test)

#---------- Convert ----------# test$SalePrice to an integer and row bind Test to Train
test$SalePrice <- as.integer(test$SalePrice)

data_full <- bind_rows(train, test)

# Inspect data_full
introduce(data_full)
str(data_full)
plot_missing(data_full)
plot_histogram(data_full)

# Iteratively improve data_full


#---------- Square Footage variables ----------#

# Create 
square_footage <- data_full %>%
  rename(FirstFloorSF = X1stFlrSF,
         SecondFloorSF = X2ndFlrSF,
         ThreeSeasonPorch = X3SsnPorch) %>%
  select(LotFrontage, LotArea, MasVnrArea, BsmtFinSF1,
         BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, FirstFloorSF,
         SecondFloorSF, LowQualFinSF, GrLivArea, GarageArea,
         WoodDeckSF, OpenPorchSF, EnclosedPorch, ThreeSeasonPorch,
         ScreenPorch, PoolArea, SalePrice, dataPartition) %>%
  mutate(NewTotBsmtSF = BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF,
         NewTotGrLivArea = FirstFloorSF + SecondFloorSF + LowQualFinSF,
         StructureSF = TotalBsmtSF + GrLivArea + GarageArea + WoodDeckSF + 
           OpenPorchSF + EnclosedPorch + ThreeSeasonPorch + ScreenPorch + 
           PoolArea)

# Visualize relationship between SF variables
ggplot(square_footage, aes(NewTotBsmtSF, TotalBsmtSF)) +
  geom_point()
ggplot(square_footage, aes(NewTotGrLivArea, GrLivArea)) +
  geom_point()
ggplot(square_footage, aes(TotalBsmtSF, GrLivArea)) +
  geom_point()

pairs(square_footage)

# Inspect square_footage
introduce(square_footage)
str(square_footage)
plot_missing(square_footage)
multi.hist(square_footage, breaks = 100)

# Inspect GarageArea
ggplot(data_full, aes(GarageArea)) +
  geom_histogram(binwidth = 50)
summary(data_full$GarageArea)

# Impute median GarageArea for the single NA
median(data_full$GarageArea, na.rm = T)
data_full$GarageArea[is.na(data_full$GarageArea)] <- 480

# Inspect TotalBsmtSF
ggplot(data_full, aes(TotalBsmtSF)) +
  geom_histogram(binwidth = 50)

summary(data_full$TotalBsmtSF)
median(data_full$TotalBsmtSF, na.rm = T)

data_full$TotalBsmtSF[is.na(data_full$TotalBsmtSF)] <- 989.5

# Inspect BsmtUnfSF
ggplot(data_full, aes(BsmtUnfSF)) +
  geom_histogram(binwidth = 50)

summary(data_full$BsmtUnfSF)
median(data_full$BsmtUnfSF, na.rm = T)

data_full$BsmtUnfSF[is.na(data_full$BsmtUnfSF)] <- 467

# Inspect BsmtFinSF1
ggplot(data_full, aes(BsmtFinSF1)) +
  geom_histogram(binwidth = 50)

summary(data_full$BsmtFinSF1)
median(data_full$BsmtFinSF1, na.rm = T)

data_full$BsmtFinSF1[is.na(data_full$BsmtFinSF1)] <- 368.5

# Inspect BsmtFinSF2
ggplot(data_full, aes(BsmtFinSF2)) +
  geom_histogram(binwidth = 50)

summary(data_full$BsmtFinSF2)
median(data_full$BsmtFinSF2, na.rm = T)

data_full$BsmtFinSF2[is.na(data_full$BsmtFinSF2)] <- 49.58

# Inspect MasVnrArea
ggplot(data_full, aes(MasVnrArea)) +
  geom_histogram(binwidth = 50)

summary(data_full$MasVnrArea)
median(data_full$MasVnrArea, na.rm = T)

data_full$MasVnrArea[is.na(data_full$MasVnrArea)] <- 102.2

# Inspect LotFrontage
ggplot(data_full, aes(LotFrontage)) +
  geom_histogram(binwidth = 50)

summary(data_full$LotFrontage)
median(data_full$LotFrontage, na.rm = T)

data_full$LotFrontage[is.na(data_full$LotFrontage)] <- 68

# Create StructureSF variable in data_full
data_full <- data_full %>%
  rename(FirstFloorSF = X1stFlrSF,
         SecondFloorSF = X2ndFlrSF,
         ThreeSeasonPorch = X3SsnPorch) %>%
  mutate(StructureSF = TotalBsmtSF + GrLivArea + GarageArea + WoodDeckSF + 
           OpenPorchSF + EnclosedPorch + ThreeSeasonPorch + ScreenPorch + 
           PoolArea)

# Visualize StructureSF
ggplot(data_full, aes(StructureSF)) +
  geom_histogram(binwidth = 250, fill = "#2c3539", col = "#a3c1ad") +
  ylab("Count")

ggplot(data_full, aes(StructureSF, SalePrice)) +
  geom_point(col = "#2c3539", size = 2) +
  stat_smooth(method = "loess", col = "#DC143C") +
  scale_y_continuous(labels = scales::dollar)

#---------- Count variables ----------#
count_variables <- data_full %>%
  select(BsmtFullBath, BsmtHalfBath, FullBath, HalfBath,
         BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, Fireplaces,
         GarageCars, SalePrice, dataPartition) %>%
  mutate(TotalBathrooms = BsmtFullBath + BsmtHalfBath + FullBath + HalfBath,
         TotalRooms = TotalBathrooms + BedroomAbvGr + KitchenAbvGr,
         FeatureCount = TotalRooms + Fireplaces + GarageCars)

ggplot(count_variables, aes(TotalRooms, TotRmsAbvGrd)) +
  geom_point()
summary(count_variables$TotalRooms)
summary(count_variables$TotRmsAbvGrd)

ggplot(count_variables, aes(FeatureCount)) +
  geom_bar()

ggplot(count_variables, aes(FeatureCount, SalePrice)) +
  geom_point(col = "#2c3539", size = 2) +
  stat_smooth(method = "loess", col = "#DC143C") +
  scale_y_continuous(labels = scales::dollar)

pairs(count_variables[, c("SalePrice", "TotalBathrooms", "TotalRooms", "FeatureCount")])

# Inspect count_variables
introduce(count_variables)
str(count_variables)
plot_missing(count_variables)
plot_histogram(count_variables)
multi.hist(count_variables)

# Inspect GarageCars
ggplot(data_full, aes(GarageCars)) +
  geom_bar()

summary(data_full$GarageCars)
median(data_full$GarageCars, na.rm = T)

data_full$GarageCars[is.na(data_full$GarageCars)] <- 2

# Inspect BsmtHalfBath
ggplot(data_full, aes(BsmtHalfBath)) +
  geom_bar()
table(data_full$BsmtHalfBath)

summary(data_full$BsmtHalfBath)

data_full$BsmtHalfBath[is.na(data_full$BsmtHalfBath)] <- 0

# Inspect BsmtFullBath
ggplot(data_full, aes(BsmtFullBath)) +
  geom_bar()
table(data_full$BsmtFullBath)

summary(data_full$BsmtFullBath)

data_full$BsmtFullBath[is.na(data_full$BsmtFull)] <- 0

# Create FeatureCount, TotalRooms, TotalBathrooms variable(s) in data_full
data_full <- data_full %>%
  mutate(TotalBathrooms = BsmtFullBath + BsmtHalfBath + FullBath + HalfBath,
         TotalRooms = TotalBathrooms + BedroomAbvGr + KitchenAbvGr,
         FeatureCount = TotalRooms + Fireplaces + GarageCars)

# Visualize FeatureCount
pairs(data_full[, c("SalePrice", "StructureSF", "FeatureCount")])

ggplot(data_full, aes(FeatureCount)) +
  geom_bar(fill = "#2c3539", col = "#a3c1ad") +
  ylab("Count")

ggplot(data_full, aes(FeatureCount, SalePrice)) +
  geom_point(fill = "#2c3539", col = "#002147") +
  stat_smooth(method = "loess", col = "#DC143C")
  ylab("Sale Price") +
  scale_y_continuous(labels = scales::dollar)

#---------- Dollar variables ----------#
dollar_variables <- data_full %>%
  select(MiscVal, SalePrice) %>%
  filter(MiscVal > 0)

ggplot(dollar_variables, aes(MiscVal)) +
  geom_histogram(binwidth = 1000)

ggplot(dollar_variables, aes(MiscVal, SalePrice)) +
  geom_point()

# Inspect dollar_variables
introduce(dollar_variables)
str(dollar_variables)
plot_missing(dollar_variables)
plot_histogram(dollar_variables)

#---------- Categorical / Ordinal ----------#
ordinal_variables <- data_full %>%
  select(MSSubClass, OverallQual, OverallCond)

# Inspect ordinal_variables
introduce(ordinal_variables)
str(ordinal_variables)
plot_histogram(ordinal_variables)

# Inspect MSSubClass
ggplot(ordinal_variables, aes(MSSubClass)) +
  geom_bar()

ordinal_variables$MSSubClass <- as.factor(ordinal_variables$MSSubClass)

# Inspect OverallQual
ggplot(ordinal_variables, aes(OverallQual)) +
  geom_bar()

ordinal_variables <- ordinal_variables %>%
  mutate(OverallQual = ordered(OverallQual, levels = c(1:10)))

str(ordinal_variables)

# Inspect OverallCond
ggplot(ordinal_variables, aes(OverallCond)) +
  geom_bar()

ordinal_variables <- ordinal_variables %>%
  mutate(OverallCond <- ordered(OverallCond, levels = c(1:10)))

# Update data_full with categorical / ordinal fixes
data_full$MSSubClass <- as.factor(data_full$MSSubClass)
data_full <- data_full %>%
  mutate(OverallQual = ordered(OverallQual, levels = c(1:10)),
         OverallCond = ordered(OverallCond, levels = c(1:10)))

#---------- Missing Values ----------#
missing_variables <- data_full %>%
  select(SaleType, KitchenQual, Electrical, Exterior2nd,
         Exterior1st, Functional, Utilities, MSZoning,
         MasVnrType, BsmtFinType1, BsmtFinType2, BsmtQual,
         BsmtExposure, BsmtCond, GarageType, GarageCond,
         GarageQual, GarageFinish, GarageYrBlt, FireplaceQu,
         Fence, Alley, MiscFeature, PoolQC)

# Inspect missing_variables
plot_missing(missing_variables)
str(missing_variables)

# Inspect SaleType
ggplot(missing_variables, aes(SaleType)) +
  geom_bar()

table(missing_variables$SaleType)
summary(missing_variables$SaleType)

data_full$SaleType[is.na(data_full$SaleType)] <- "WD"

# Inspect Exterior1st
ggplot(missing_variables, aes(Exterior1st)) +
  geom_bar()

table(missing_variables$Exterior1st)

data_full$Exterior1st[is.na(data_full$Exterior1st)] <- "VinylSd"
data_full$Exterior1st <- as.factor(data_full$Exterior1st)

# Inspect Exterior2nd
ggplot(missing_variables, aes(Exterior2nd)) +
  geom_bar()

table(missing_variables$Exterior2nd)

data_full$Exterior2nd[is.na(data_full$Exterior2nd)] <- "VinylSd"
data_full$Exterior2nd <- as.factor(data_full$Exterior2nd)

# Inspect Electrical
ggplot(missing_variables, aes(Electrical)) +
  geom_bar()

table(missing_variables$Electrical)

data_full$Electrical[is.na(data_full$Electrical)] <- "SBrkr"

# Inspect KitchenQual
ggplot(missing_variables, aes(KitchenQual)) +
  geom_bar()

table(missing_variables$KitchenQual)

data_full$KitchenQual <- ifelse(data_full$KitchenQual == "Po", "NA",
                                ifelse(data_full$KitchenQual == "FA", "2",
                                       ifelse(data_full$KitchenQual == "TA", "3",
                                              ifelse(data_full$KitchenQual == "Gd", "4",
                                                     ifelse(data_full$KitchenQual == "Ex", "5", "NA")))))

data_full$KitchenQual[is.na(data_full$KitchenQual)] <- "TA"
data_full$KitchenQual <- as.numeric(data_full$KitchenQual)

ggplot(data_full, aes(KitchenQual)) +
  geom_bar()

table(data_full$KitchenQual)







# Missing Values
train$MasVnrArea[is.na(train$MasVnrArea)] <- 0
train$LotFrontage[is.na(train$LotFrontage)] <- 0

# Dataset Build

# Neighborhood Values
median_neighborhood_value <- train %>%
  select(Neighborhood, SalePrice) %>%
  group_by(Neighborhood) %>%
  summarise(Median_SalePrice = median(SalePrice)) %>%
  mutate(Neighborhood_Values = cut(Median_SalePrice, breaks = 10, labels = c(1:10))) %>%
  select(Neighborhood, Neighborhood_Values)

train <- left_join(train, median_neighborhood_value, by.x = "Neighborhood", by.y = "Neighborhood")

# Foundation Type
train$Foundation_Type <- ifelse(train$Foundation == "BrkTil", "1",
                                ifelse(train$Foundation == "Slab", "1",
                                       ifelse(train$Foundation == "Stone", "1",
                                              ifelse(train$Foundation == "Wood", "1",
                                                     ifelse(train$Foundation == "CBlock", "2",
                                                            ifelse(train$Foundation == "PConc", "3", "NA"))))))

train.NA.rm <- train %>%
  rename(FirstFloorSF = X1stFlrSF, 
         SecondFloorSF = X2ndFlrSF,
         Bedroom = BedroomAbvGr, 
         Kitchen = KitchenAbvGr,
         ThreeSeasonPorch = X3SsnPorch,
         Rooms = TotRmsAbvGrd) %>%
  select(SalePrice, LotArea, FirstFloorSF, SecondFloorSF, Bedroom, 
         Rooms, GarageArea, BsmtUnfSF, FullBath, HalfBath, OverallQual,
         BsmtFinSF1, BsmtFullBath, BsmtHalfBath, YearBuilt, YrSold, 
         GrLivArea, Neighborhood, MSSubClass, Fireplaces, TotalBsmtSF,
         Neighborhood_Values, Foundation_Type) %>%
  mutate(SalePrice_Z = (SalePrice - mean(SalePrice))/sd(SalePrice),
         LotArea_Z = (LotArea - mean(LotArea))/sd(LotArea),
         FirstFloorSF_Z = (FirstFloorSF - mean(FirstFloorSF))/sd(FirstFloorSF),
         GrLivArea_Z = (GrLivArea - mean(GrLivArea))/sd(GrLivArea),
         GarageArea_Z = (GarageArea - mean(GarageArea))/sd(GarageArea),
         BsmtBath = BsmtFullBath + BsmtHalfBath,
         GrBath = FullBath + HalfBath,
         Bath = BsmtBath + GrBath,
         DecadeBuilt = cut(YearBuilt, breaks = 14, labels = c(1:14)),
         BasementSF = BsmtFinSF1 + BsmtUnfSF,
         AbvGrSF = FirstFloorSF + SecondFloorSF,
         HouseSF = AbvGrSF + TotalBsmtSF,
         HouseSF_Z = (HouseSF - mean(HouseSF))/sd(HouseSF),
         BasementSF_Z = (BasementSF - mean(BasementSF))/sd(BasementSF),
         DecadeBuilt = as.numeric(DecadeBuilt),
         Age = (YrSold - YearBuilt) + 1,
         MSSubClass = as.factor(MSSubClass),
         Neighborhood_Values = as.numeric(Neighborhood_Values),
         Foundation_Type = as.numeric(Foundation_Type)) %>%
  mutate(SalePrice = log10(SalePrice),
         OverallQual = log10(OverallQual),
         LotArea = log10(LotArea),
         FirstFloorSF = log10(FirstFloorSF),
         GrLivArea = log10(GrLivArea),
         GarageArea = log10(GarageArea + 1),
         Rooms = log10(Rooms),
         Bath = log10(Bath),
         BasementSF = log10(BasementSF),
         HosueSF = log10(HouseSF),
         Fireplaces = log10(Fireplaces),
         Age = log10(Age + 1),
         Neighborhood_Values = log10(Neighborhood_Values)) %>%
  select(SalePrice, OverallQual, LotArea, GarageArea, 
         Rooms, Bath, HouseSF, Age, Neighborhood_Values)


# filter(SalePrice_Z >= -2.5 & SalePrice_Z <= 2.5,
#        LotArea_Z >= -2.5 & LotArea_Z <= 2.5,
#        FirstFloorSF_Z >= -2.5 & FirstFloorSF_Z <= 2.5,
#        GrLivArea_Z >= -2.5 & GrLivArea_Z <= 2.5,
#        GarageArea_Z >= -2.5 & GarageArea_Z <= 2.5,
#        BasementSF_Z >= -2.5 & BasementSF_Z <= 2.5,
#        HouseSF_Z >= -2.5 & HouseSF_Z <= 2.5)  %>%

str(train)

# Exploratory Data Analysis
introduce(train.NA.rm)
str(train.NA.rm)
plot_missing(train.NA.rm)
pairs(train.NA.rm)

plot_bar(train.NA.rm,
         theme_config = list("rect" = element_blank(),
                             "text" = element_text(colour = "#2c3539", size = 12),
                             "line" = element_blank(),
                             "axis.text.x" = element_text(angle = 45, hjust = 1,
                                                          colour = "#2c3539", size = 12),
                             "axis.text.y" = element_text(colour = "#2c3539", size = 12),
                             "axis.line.y" = element_line(colour = "#2c3539", size = .7),
                             "axis.line.x" = element_line(colour = "#2c3539", size = .7),
                             "axis.title" = element_text(colour = "#2c3539", size = 12)))

plot_histogram(train.NA.rm, 
               theme_config = list("rect" = element_blank(),
                                   "text" = element_text(colour = "#2c3539", size = 12),
                                   "line" = element_blank(),
                                   "axis.text.x" = element_text(angle = 45, hjust = 1,
                                                                colour = "#2c3539", size = 12),
                                   "axis.text.y" = element_text(colour = "#2c3539", size = 12),
                                   "axis.line.y" = element_line(colour = "#2c3539", size = .7),
                                   "axis.line.x" = element_line(colour = "#2c3539", size = .7),
                                   "axis.title" = element_text(colour = "#2c3539", size = 12)))

plot_boxplot(train.NA.rm,
             by = "SalePrice",
             theme_config = list("rect" = element_blank(),
                                 "text" = element_text(colour = "#2c3539", size = 12),
                                 "line" = element_blank(),
                                 "axis.text.x" = element_text(angle = 45, hjust = 1,
                                                              colour = "#2c3539", size = 12),
                                 "axis.text.y" = element_text(colour = "#2c3539", size = 12),
                                 "axis.line.y" = element_line(colour = "#2c3539", size = .7),
                                 "axis.line.x" = element_line(colour = "#2c3539", size = .7),
                                 "axis.title" = element_text(colour = "#2c3539", size = 12)))

plot_scatterplot(train.NA.rm,
                 by = "SalePrice",
                 theme_config = list("rect" = element_blank(),
                                     "text" = element_text(colour = "#2c3539", size = 12),
                                     "line" = element_blank(),
                                     "axis.text.x" = element_text(angle = 45, hjust = 1,
                                                                  colour = "#2c3539", size = 12),
                                     "axis.text.y" = element_text(colour = "#2c3539", size = 12),
                                     "axis.line.y" = element_line(colour = "#2c3539", size = .7),
                                     "axis.line.x" = element_line(colour = "#2c3539", size = .7),
                                     "axis.title" = element_text(colour = "#2c3539", size = 12)))

plot_correlation(train.NA.rm,
                 use = "pairwise.complete.obs",
                 theme_config = list("rect" = element_blank(),
                                     "text" = element_text(colour = "#2c3539", size = 12),
                                     "line" = element_blank(),
                                     "axis.text.x" = element_text(angle = 45, hjust = 1,
                                                                  colour = "#2c3539", size = 12),
                                     "axis.text.y" = element_text(colour = "#2c3539", size = 12),
                                     "axis.line.y" = element_line(colour = "#2c3539", size = .7),
                                     "axis.line.x" = element_line(colour = "#2c3539", size = .7),
                                     "axis.title" = element_text(colour = "#2c3539", size = 12)))

# Boruta Feature Selection
set.seed(332211)
boruta_train <- Boruta(SalePrice ~ ., data = train.NA.rm, doTrace = 2, ntree = 500)

plot(boruta_train)
str(boruta_train)

# Linear Model
str(train.NA.rm)
plot_missing(train.NA.rm)
lm_2 <- lm(SalePrice ~., data = train.NA.rm)
summary(lm_2)
pairs(train.NA.rm)

#--------------------------------------------------------------------------#
#                             Model Building                               #
#--------------------------------------------------------------------------#


#                         Ensemble - SVM, RF, GBM                           #

train.1 <- train.NA.rm
train.label <- train.NA.rm$SalePrice
train.1$SalePrice <- NULL

# DummyVars
# dmy.train <- dummyVars(" ~ .", data = train.1)
# train.1 <- data.frame(predict(dmy.train, newdata = train.1))

# Create Folds Index for CV
set.seed(394858)
cv_1_folds_1 <- createMultiFolds(train.label,
                                 k = 10,
                                 times = 10)
set.seed(1234)
objControl <- trainControl(method = "repeatedcv",
                           number = 10, 
                           repeats = 10,
                           search = "grid",
                           savePredictions = T,
                           index = cv_1_folds_1,
                           verboseIter = T)

# Parameter Grids
GBM.grid<- expand.grid(interaction.depth = 8,
                       shrinkage = 0.01,
                       n.trees = 1500,
                       n.minobsinnode = 2)

RF.grid<- expand.grid(mtry = c(1, 2, 4, 8, 16))

SVM.grid <- expand.grid(C = c(0.25, .5, .75, 1))

# Models
time1 <- Sys.time()

set.seed(56568)
GBM.model <- train(x = train.1,
                   y = train.label,
                   trControl = objControl,
                   metric = "RMSE",
                   method = "gbm",
                   tuneGrid = GBM.grid,
                   preProcess = c("center", "scale"))

summary(GBM.model)

set.seed(989898)                   
RF.model <- train(x = train.1,
                  y = train.label,
                  trControl = objControl,
                  metric = "RMSE",
                  method = "rf",
                  tuneGrid = RF.grid,
                  tuneLength = 15,
                  ntree = 1000)

summary(RF.model)

set.seed(989898)                   
SVM.model <- train(x = train.1,
                  y = train.label,
                  trControl = objControl,
                  metric = "RMSE",
                  method = "svmLinear",
                  tuneGrid = SVM.grid)

time2 <- Sys.time()
elapsedTime <- time2 - time1
elapsedTime


GBM.model
RF.model
SVM.model

# Evaluate
combined_models<- list(gbm = GBM.model, rf = RF.model, svmLinear = SVM.model)
class(combined_models) <- "caretList"

modelCor(resamples(combined_models))
summary(resamples(combined_models))

# Weighted Ensemble
set.seed(993311)
models_ensemble <- caretEnsemble(combined_models)

# Evaluate
summary(models_ensemble)





# Test set build
plot_missing(test)

test.NA <- test %>%
  rename(FirstFloorSF = X1stFlrSF, 
         SecondFloorSF = X2ndFlrSF,
         Bedroom = BedroomAbvGr, 
         Kitchen = KitchenAbvGr,
         ThreeSeasonPorch = X3SsnPorch,
         Rooms = TotRmsAbvGrd) %>%
  select(LotArea, FirstFloorSF, SecondFloorSF, Bedroom, Rooms, GarageArea, BsmtUnfSF, Neighborhood,
         BsmtFinSF1, BsmtFullBath, BsmtHalfBath, YearBuilt, YrSold, GrLivArea, TotalBsmtSF)

plot_missing(test.NA)

# BsmtFinSF1
mean(test$BsmtFinSF1, na.rm = T)
test$BsmtFinSF1[is.na(test$BsmtFinSF1)] <- mean(test$BsmtFinSF1, na.rm = T)
mean(test$BsmtFinSF1)

# BsmtUnfSF
mean(test$BsmtUnfSF, na.rm = T)
test$BsmtUnfSF[is.na(test$BsmtUnfSF)] <- mean(test$BsmtUnfSF, na.rm = T)
mean(test$BsmtUnfSF)

# GarageArea
mean(test$GarageArea, na.rm = T)
test$GarageArea[is.na(test$GarageArea)] <- mean(test$GarageArea, na.rm = T)
mean(test$GarageArea)

# BsmtHalfBath
mean(test$BsmtHalfBath, na.rm = T)
test$BsmtHalfBath[is.na(test$BsmtHalfBath)] <- mean(test$BsmtHalfBath, na.rm = T)
mean(test$BsmtHalfBath)

# BsmtFullBath
mean(test$BsmtFullBath, na.rm = T)
test$BsmtFullBath[is.na(test$BsmtFullBath)] <- mean(test$BsmtFullBath, na.rm = T)
mean(test$BsmtFullBath)

# TotBsmtSF
mean(test$TotalBsmtSF, na.rm = T)
test$TotalBsmtSF[is.na(test$TotalBsmtSF)] <- mean(test$TotalBsmtSF, na.rm = T)
mean(test$TotalBsmtSF)


# Neighborhood Values
test <- left_join(test, median_neighborhood_value, by.x = "Neighborhood", by.y = "Neighborhood")

# Foundation Type
levels(as.factor(test$Foundation))
test$Foundation_Type <- ifelse(test$Foundation == "BrkTil", "1",
                                ifelse(test$Foundation == "Slab", "1",
                                       ifelse(test$Foundation == "Stone", "1",
                                              ifelse(test$Foundation == "Wood", "1",
                                                     ifelse(test$Foundation == "CBlock", "2",
                                                            ifelse(test$Foundation == "PConc", "3", "NA"))))))

test.NA.rm <- test %>%
  rename(FirstFloorSF = X1stFlrSF, 
         SecondFloorSF = X2ndFlrSF,
         Bedroom = BedroomAbvGr, 
         Kitchen = KitchenAbvGr,
         ThreeSeasonPorch = X3SsnPorch,
         Rooms = TotRmsAbvGrd) %>%
  select(LotArea, FirstFloorSF, SecondFloorSF, Bedroom, 
         Rooms, GarageArea, BsmtUnfSF, FullBath, 
         HalfBath, OverallQual, BsmtFinSF1, BsmtFullBath, 
         BsmtHalfBath, YearBuilt, YrSold, GrLivArea, 
         Neighborhood, MSSubClass, Fireplaces, TotalBsmtSF, 
         Neighborhood_Values, Foundation_Type) %>%
  mutate(LotArea_Z = (LotArea - mean(LotArea))/sd(LotArea),
         FirstFloorSF_Z = (FirstFloorSF - mean(FirstFloorSF))/sd(FirstFloorSF),
         GrLivArea_Z = (GrLivArea - mean(GrLivArea))/sd(GrLivArea),
         GarageArea_Z = (GarageArea - mean(GarageArea))/sd(GarageArea),
         BsmtBath = BsmtFullBath + BsmtHalfBath,
         GrBath = FullBath + HalfBath,
         Bath = BsmtBath + GrBath,
         DecadeBuilt = cut(YearBuilt, breaks = 14, labels = c(1:14)),
         BasementSF = BsmtFinSF1 + BsmtUnfSF,
         AbvGrSF = FirstFloorSF + SecondFloorSF,
         HouseSF = AbvGrSF + TotalBsmtSF,
         HouseSF_Z = (HouseSF - mean(HouseSF))/sd(HouseSF),
         BasementSF_Z = (BasementSF - mean(BasementSF))/sd(BasementSF),
         DecadeBuilt = as.numeric(DecadeBuilt),
         Age = (YrSold - YearBuilt) + 1,
         MSSubClass = as.factor(MSSubClass),
         Neighborhood_Values = as.numeric(Neighborhood_Values),
         Foundation_Type = as.numeric(Foundation_Type)) %>%
  mutate(OverallQual = log10(OverallQual),
         LotArea = log10(LotArea),
         FirstFloorSF = log10(FirstFloorSF),
         GrLivArea = log10(GrLivArea),
         GarageArea = log10(GarageArea + 1),
         Rooms = log10(Rooms),
         Bath = log10(Bath),
         BasementSF = log10(BasementSF),
         HosueSF = log10(HouseSF),
         Fireplaces = log10(Fireplaces),
         Age = log10(Age + 1),
         Neighborhood_Values = log10(Neighborhood_Values))  %>%
  select(OverallQual, LotArea, GarageArea, 
         Rooms, Bath, HouseSF, Age, Neighborhood_Values)

plot_missing(test.NA.rm)
test.NA.rm$Age[is.na(test.NA.rm$Age)] <- round(mean(test.NA.rm$Age, na.rm = T), 0)
test.NA.rm$HouseSF[is.na(test.NA.rm$HouseSF)] <- round(mean(test.NA.rm$HouseSF, na.rm = T), 0)
test.NA.rm$GarageArea[is.na(test.NA.rm$GarageArea)] <- round(mean(test.NA.rm$GarageArea, na.rm = T), 0)
test.NA.rm$Bath[is.na(test.NA.rm$Bath)] <- round(mean(test.NA.rm$Bath, na.rm = T), 0)

pairs(test.NA.rm)
plot_correlation(test.NA.rm)
str(test.NA.rm)
introduce(test.NA.rm)
introduce(train.NA.rm)

# DummyVars
# dmy.test <- dummyVars(" ~ .", data = test.NA.rm)
# test.NA.rm <- data.frame(predict(dmy.train, newdata = test.NA.rm))


# Predictions and Submission

preds_rf <- predict(models_ensemble, test.NA.rm)
str(preds_rf)

log_submission <- data.frame(Id = test$Id, log_price = preds_rf)

gbm_submission <- log_submission %>%
  mutate(SalePrice = 10^log_price) %>%
  select(Id, SalePrice)

head(gbm_submission)

write.csv(gbm_submission, file = "submission_ensemble.csv")












#                             Random Forest                                #
train.1 <- train.NA.rm
train.label <- train.NA.rm$SalePrice
train.1$SalePrice <- NULL

set.seed(1234)
cv_10_folds <- createMultiFolds(train.label,
                                k = 10,
                                times = 10)

objControl <- trainControl(method = "repeatedcv",
                           number = 10, 
                           repeats = 10,
                           index = cv_10_folds)

time1 <- Sys.time()
c1 <- makeCluster(4, type = "SOCK")
registerDoSNOW(c1)

rf_1_cv_1 <- train(x = train.1,
                   y = train.label,
                   method = "rf",
                   tuneLength = 3,
                   ntree = 1000,
                   trControl = objControl,
                   importance = T)

stopCluster(c1)
time2 <- Sys.time()
plot(varImp(rf_1_cv_1))
summary(rf_1_cv_1)
elapsedTime <- time2 - time1
elapsedTime



# Test set build
plot_missing(test)

test.NA <- test %>%
  rename(FirstFloorSF = X1stFlrSF, 
         SecondFloorSF = X2ndFlrSF,
         Bedroom = BedroomAbvGr, 
         Kitchen = KitchenAbvGr,
         ThreeSeasonPorch = X3SsnPorch,
         Rooms = TotRmsAbvGrd) %>%
  select(LotArea, FirstFloorSF, SecondFloorSF, Bedroom, Rooms, GarageArea, BsmtUnfSF,
         BsmtFinSF1, BsmtFullBath, BsmtHalfBath, YearBuilt, YrSold, GrLivArea, TotalBsmtSF)

plot_missing(test.NA)

# BsmtFinSF1
mean(test$BsmtFinSF1, na.rm = T)
test$BsmtFinSF1[is.na(test$BsmtFinSF1)] <- mean(test$BsmtFinSF1, na.rm = T)
mean(test$BsmtFinSF1)

# BsmtUnfSF
mean(test$BsmtUnfSF, na.rm = T)
test$BsmtUnfSF[is.na(test$BsmtUnfSF)] <- mean(test$BsmtUnfSF, na.rm = T)
mean(test$BsmtUnfSF)

# GarageArea
mean(test$GarageArea, na.rm = T)
test$GarageArea[is.na(test$GarageArea)] <- mean(test$GarageArea, na.rm = T)
mean(test$GarageArea)

# BsmtHalfBath
mean(test$BsmtHalfBath, na.rm = T)
test$BsmtHalfBath[is.na(test$BsmtHalfBath)] <- mean(test$BsmtHalfBath, na.rm = T)
mean(test$BsmtHalfBath)

# BsmtFullBath
mean(test$BsmtFullBath, na.rm = T)
test$BsmtFullBath[is.na(test$BsmtFullBath)] <- mean(test$BsmtFullBath, na.rm = T)
mean(test$BsmtFullBath)

# TotBsmtSF
mean(test$TotalBsmtSF, na.rm = T)
test$TotalBsmtSF[is.na(test$TotalBsmtSF)] <- mean(test$TotalBsmtSF, na.rm = T)
mean(test$TotalBsmtSF)

plot_missing(test.NA)

test.NA.rm <- test %>%
  rename(FirstFloorSF = X1stFlrSF, 
         SecondFloorSF = X2ndFlrSF,
         Bedroom = BedroomAbvGr, 
         Kitchen = KitchenAbvGr,
         ThreeSeasonPorch = X3SsnPorch,
         Rooms = TotRmsAbvGrd) %>%
  select(LotArea, FirstFloorSF, SecondFloorSF, Bedroom, 
         Rooms, GarageArea, BsmtUnfSF, FullBath, HalfBath, OverallQual,
         BsmtFinSF1, BsmtFullBath, BsmtHalfBath, YearBuilt, YrSold, 
         GrLivArea, Neighborhood, MSSubClass, Fireplaces, TotalBsmtSF) %>%
  mutate(LotArea_Z = (LotArea - mean(LotArea))/sd(LotArea),
         FirstFloorSF_Z = (FirstFloorSF - mean(FirstFloorSF))/sd(FirstFloorSF),
         GrLivArea_Z = (GrLivArea - mean(GrLivArea))/sd(GrLivArea),
         GarageArea_Z = (GarageArea - mean(GarageArea))/sd(GarageArea),
         BsmtBath = BsmtFullBath + BsmtHalfBath,
         GrBath = FullBath + HalfBath,
         Bath = BsmtBath + GrBath,
         DecadeBuilt = cut(YearBuilt, breaks = 14, labels = c(1:14)),
         BasementSF = BsmtFinSF1 + BsmtUnfSF,
         AbvGrSF = FirstFloorSF + SecondFloorSF,
         HouseSF = AbvGrSF + TotalBsmtSF,
         HouseSF_Z = (HouseSF - mean(HouseSF))/sd(HouseSF),
         BasementSF_Z = (BasementSF - mean(BasementSF))/sd(BasementSF),
         DecadeBuilt = as.numeric(DecadeBuilt),
         Age = YrSold - YearBuilt,
         MSSubClass = as.factor(MSSubClass)) %>%
  select(OverallQual, LotArea, FirstFloorSF,
         GrLivArea, GarageArea, Rooms, Bath, BasementSF,
         HouseSF, Fireplaces, Age)

plot_missing(test.NA.rm)
introduce(test.NA.rm)
introduce(train.NA.rm)

preds_rf <- predict(rf_1_cv_1, test.NA.rm)
str(preds_rf)

rf_submission <- data.frame(Id = test$Id, SalePrice = preds_rf)
write.csv(rf_submission, file = "Test.Submission.RF.Continuous.csv")

#                             GBM                                #

train.1 <- train.NA.rm
train.label <- train.NA.rm$SalePrice
train.1$SalePrice <- NULL

# DummyVars
dmy.train <- dummyVars(" ~ .", data = train.1)
train.1 <- data.frame(predict(dmy.train, newdata = train.1))

set.seed(1234)
cv_10_folds <- createMultiFolds(train.label,
                                k = 10,
                                times = 10)

objControl <- trainControl(method = "repeatedcv",
                           number = 10, 
                           repeats = 10,
                           index = cv_10_folds)

# Parameters
grid.search <- expand.grid(interaction.depth = c(1, 2, 4, 8),
                           shrinkage = c(0.1, 0.01, 0.001),
                           n.trees = c(500, 1000, 1500, 2000),
                           n.minobsinnode = c(1, 2, 4, 8))

grid.search <- expand.grid(interaction.depth = 8,
                           shrinkage = 0.01,
                           n.trees = 1500,
                           n.minobsinnode = 2)

# Model train
time1 <- Sys.time()
c1 <- makeCluster(4, type = "SOCK")
registerDoSNOW(c1)

gbm.1.cv.1 <- train(x = train.1,
                   y = train.label,
                   method = "gbm",
                   trControl = objControl,
                   tuneGrid = grid.search,
                   preProcess = c("center", "scale"))

stopCluster(c1)
time2 <- Sys.time()
elapsedTime <- time2 - time1
elapsedTime
plot(varImp(gbm.1.cv.1))
gbm.1.cv.1



# Evaluate
plot.gbm(gbm.1.cv.1)
str(gbm.1.cv.1$results)
gbm_data <- as.data.frame(gbm.1.cv.1$results)
str(gbm_data)


bin.number <- nclass.FD(gbm_data$RMSE)
bin.width.RMSE <- max(gbm_data$RMSE)/bin.number
ggplot(gbm_data, aes(RMSE)) +
  geom_histogram(binwidth = bin.width.RMSE)

ggplot(gbm_data, aes(shrinkage, RMSE)) +
  geom_point()





# Test set build
plot_missing(test)

test.NA <- test %>%
  rename(FirstFloorSF = X1stFlrSF, 
         SecondFloorSF = X2ndFlrSF,
         Bedroom = BedroomAbvGr, 
         Kitchen = KitchenAbvGr,
         ThreeSeasonPorch = X3SsnPorch,
         Rooms = TotRmsAbvGrd) %>%
  select(LotArea, FirstFloorSF, SecondFloorSF, Bedroom, Rooms, GarageArea, BsmtUnfSF, Neighborhood,
         BsmtFinSF1, BsmtFullBath, BsmtHalfBath, YearBuilt, YrSold, GrLivArea, TotalBsmtSF)

plot_missing(test.NA)

# BsmtFinSF1
mean(test$BsmtFinSF1, na.rm = T)
test$BsmtFinSF1[is.na(test$BsmtFinSF1)] <- mean(test$BsmtFinSF1, na.rm = T)
mean(test$BsmtFinSF1)

# BsmtUnfSF
mean(test$BsmtUnfSF, na.rm = T)
test$BsmtUnfSF[is.na(test$BsmtUnfSF)] <- mean(test$BsmtUnfSF, na.rm = T)
mean(test$BsmtUnfSF)

# GarageArea
mean(test$GarageArea, na.rm = T)
test$GarageArea[is.na(test$GarageArea)] <- mean(test$GarageArea, na.rm = T)
mean(test$GarageArea)

# BsmtHalfBath
mean(test$BsmtHalfBath, na.rm = T)
test$BsmtHalfBath[is.na(test$BsmtHalfBath)] <- mean(test$BsmtHalfBath, na.rm = T)
mean(test$BsmtHalfBath)

# BsmtFullBath
mean(test$BsmtFullBath, na.rm = T)
test$BsmtFullBath[is.na(test$BsmtFullBath)] <- mean(test$BsmtFullBath, na.rm = T)
mean(test$BsmtFullBath)

# TotBsmtSF
mean(test$TotalBsmtSF, na.rm = T)
test$TotalBsmtSF[is.na(test$TotalBsmtSF)] <- mean(test$TotalBsmtSF, na.rm = T)
mean(test$TotalBsmtSF)

plot_missing(test.NA)

test.NA.rm <- test %>%
  rename(FirstFloorSF = X1stFlrSF, 
         SecondFloorSF = X2ndFlrSF,
         Bedroom = BedroomAbvGr, 
         Kitchen = KitchenAbvGr,
         ThreeSeasonPorch = X3SsnPorch,
         Rooms = TotRmsAbvGrd) %>%
  select(LotArea, FirstFloorSF, SecondFloorSF, Bedroom, 
         Rooms, GarageArea, BsmtUnfSF, FullBath, HalfBath, OverallQual,
         BsmtFinSF1, BsmtFullBath, BsmtHalfBath, YearBuilt, YrSold, 
         GrLivArea, Neighborhood, MSSubClass, Fireplaces, TotalBsmtSF) %>%
  mutate(LotArea_Z = (LotArea - mean(LotArea))/sd(LotArea),
         FirstFloorSF_Z = (FirstFloorSF - mean(FirstFloorSF))/sd(FirstFloorSF),
         GrLivArea_Z = (GrLivArea - mean(GrLivArea))/sd(GrLivArea),
         GarageArea_Z = (GarageArea - mean(GarageArea))/sd(GarageArea),
         BsmtBath = BsmtFullBath + BsmtHalfBath,
         GrBath = FullBath + HalfBath,
         Bath = BsmtBath + GrBath,
         DecadeBuilt = cut(YearBuilt, breaks = 14, labels = c(1:14)),
         BasementSF = BsmtFinSF1 + BsmtUnfSF,
         AbvGrSF = FirstFloorSF + SecondFloorSF,
         HouseSF = AbvGrSF + TotalBsmtSF,
         HouseSF_Z = (HouseSF - mean(HouseSF))/sd(HouseSF),
         BasementSF_Z = (BasementSF - mean(BasementSF))/sd(BasementSF),
         DecadeBuilt = as.numeric(DecadeBuilt),
         Age = YrSold - YearBuilt,
         MSSubClass = as.factor(MSSubClass)) %>%
  select(OverallQual, LotArea, FirstFloorSF,
         GrLivArea, GarageArea, Rooms, Bath, BasementSF,
         HouseSF, Fireplaces, Age, Neighborhood_Values)

plot_missing(test.NA.rm)
introduce(test.NA.rm)
introduce(train.NA.rm)

# DummyVars
dmy.test <- dummyVars(" ~ .", data = test.NA.rm)
test.NA.rm <- data.frame(predict(dmy.train, newdata = test.NA.rm))

preds_rf <- predict(gbm.1.cv.1, test.NA.rm)
str(preds_rf)

gbm_submission <- data.frame(Id = test$Id, SalePrice = preds_rf)
write.csv(gbm_submission, file = "Test.Submission.GBM.Continuous.csv")











# Exploratory Plots
plot_missing(train)


# Foundation
Found.1 <- ggplot(train, aes(Foundation)) +
  geom_bar()
Found.2 <- ggplot(train, aes(Foundation, SalePrice)) +
  geom_boxplot()
grid.arrange(Found.1, Found.2, ncol = 1)

train$Foundation_Type <- ifelse(train$Foundation == "BrkTil", "3",
                                ifelse(train$Foundation == "Slab", "3",
                                   ifelse(train$Foundation == "Stone", "3",
                                          ifelse(train$Foundation == "Wood", "3",
                                                 ifelse(train$Foundation == "CBlock", "2",
                                                        ifelse(train$Foundation == "PConc", "1", "NA"))))))

ggplot(train, aes(Foundation_Type)) +
  geom_bar()
ggplot(train, aes(Foundation_Type, SalePrice)) +
  geom_boxplot()


bin.number <- nclass.FD(train$SalePrice)
bin.width.SalePrice<- max(train$SalePrice)/bin.number
ggplot(train, aes((SalePrice))) +
  geom_histogram(binwidth = bin.width.SalePrice, colour = "#D3D3D3", fill = "#2c3539") +
  scale_x_continuous(labels = scales::dollar) +
  ylab("Count") +
  theme(rect = element_blank(),
        text = element_text(colour = "#2c3539", size = 14),
        line = element_blank(),
        axis.text.x = element_text(hjust = 1, colour = "#2c3539", size = 14),
        axis.text.y = element_text(colour = "#2c3539", size = 14),
        axis.line.y = element_line(colour = "#2c3539"),
        axis.line.x = element_line(colour = "#2c3539"),
        axis.title = element_text(colour = "#2c3539", size = 14))

ggplot(train, aes(GrLivArea, SalePrice)) +
  geom_point(colour = "#002147") +
  stat_smooth(method = "lm", colour = "#a3d1ad", size = 2) +
  scale_y_continuous(labels = scales::dollar) +
  ylab("Sale Price") +
  xlab("Above Ground Living Area") +
  theme(rect = element_blank(),
        text = element_text(colour = "#2c3539", size = 14),
        line = element_blank(),
        axis.text.x = element_text(hjust = 1, colour = "#2c3539", size = 14),
        axis.text.y = element_text(colour = "#2c3539", size = 14),
        axis.line.y = element_line(colour = "#2c3539"),
        axis.line.x = element_line(colour = "#2c3539"),
        axis.title = element_text(colour = "#2c3539", size = 14))

ggplot(train, aes(group = OverallQual, OverallQual, SalePrice)) +
  geom_boxplot(fill = "#a3c1ad", colour = "#002147") +
  scale_y_continuous(labels = scales::dollar) +
  ylab("Sale Price") +
  theme(rect = element_blank(),
        text = element_text(colour = "#2c3539", size = 14),
        line = element_blank(),
        axis.text.x = element_text(hjust = 1, colour = "#2c3539", size = 14),
        axis.text.y = element_text(colour = "#2c3539", size = 14),
        axis.line.y = element_line(colour = "#2c3539"),
        axis.line.x = element_line(colour = "#2c3539"),
        axis.title = element_text(colour = "#2c3539", size = 14))

plot_SalePrice_4 <- ggplot(train, aes(group = YearBuilt, YearBuilt, SalePrice)) +
  geom_boxplot(fill = "#00ced1", colour = "#2c3539") +
  scale_y_continuous(labels = scales::dollar) +
  ylab("Sale Price") +
  theme(rect = element_blank(),
        text = element_text(colour = "#2c3539", size = 14),
        line = element_blank(),
        axis.text.x = element_text(hjust = 1, colour = "#2c3539", size = 14),
        axis.text.y = element_text(colour = "#2c3539", size = 14),
        axis.line.y = element_line(colour = "#2c3539"),
        axis.line.x = element_line(colour = "#2c3539"),
        axis.title = element_text(colour = "#2c3539", size = 14))

plot_SalePrice_5 <- ggplot(train, aes(group = Neighborhood, Neighborhood, SalePrice)) +
  geom_boxplot(fill = "#00ced1", colour = "#2c3539") +
  scale_y_continuous(labels = scales::dollar) +
  ylab("Sale Price") +
  theme(rect = element_blank(),
        text = element_text(colour = "#2c3539", size = 14),
        line = element_blank(),
        axis.text.x = element_text(angle = 75, hjust = 1, colour = "#2c3539", size = 14),
        axis.text.y = element_text(colour = "#2c3539", size = 14),
        axis.line.y = element_line(colour = "#2c3539"),
        axis.line.x = element_line(colour = "#2c3539"),
        axis.title = element_text(colour = "#2c3539", size = 14))

plot_SalePrice_6 <- ggplot(train, aes(LotArea, SalePrice)) +
  geom_point(colour = "#2c3539") +
  stat_smooth(method = "lm", colour = "#00ced1") +
  scale_y_continuous(labels = scales::dollar) +
  ylab("Sale Price") +
  xlab("Lot Area") +
  theme(rect = element_blank(),
        text = element_text(colour = "#2c3539", size = 14),
        line = element_blank(),
        axis.text.x = element_text(hjust = 1, colour = "#2c3539", size = 14),
        axis.text.y = element_text(colour = "#2c3539", size = 14),
        axis.line.y = element_line(colour = "#2c3539"),
        axis.line.x = element_line(colour = "#2c3539"),
        axis.title = element_text(colour = "#2c3539", size = 14))

multiplot(plot_SalePrice_1, plot_SalePrice_2, 
          plot_SalePrice_3, plot_SalePrice_4,
          plot_SalePrice_5, plot_SalePrice_6,
          cols = 2)

plot_SalePrice_7 <- ggplot(train, aes(GrLivArea, SalePrice)) +
  geom_point(colour = "#2c3539") +
  stat_smooth(method = "loess", colour = "#DC143C", size = 2) +
  facet_wrap(~OverallQual) +
  scale_y_continuous(labels = scales::dollar) +
  ylab("Sale Price") +
  xlab("Above Ground Living Area") +
  theme(rect = element_blank(),
        text = element_text(colour = "#2c3539", size = 10),
        line = element_blank(),
        axis.text.x = element_text(hjust = 1, colour = "#2c3539", size = 10),
        axis.text.y = element_text(colour = "#2c3539", size = 10),
        axis.line.y = element_line(colour = "#2c3539"),
        axis.line.x = element_line(colour = "#2c3539"),
        axis.title = element_text(colour = "#2c3539", size = 10))

plot_SalePrice_8 <- ggplot(train, aes(GrLivArea, SalePrice)) +
  geom_point(colour = "#2c3539") +
  stat_smooth(method = "loess", colour = "#DC143C", size = 2) +
  facet_wrap(~OverallCond) +
  scale_y_continuous(labels = scales::dollar) +
  ylab("Sale Price") +
  xlab("Above Ground Living Area") +
  theme(rect = element_blank(),
        text = element_text(colour = "#2c3539", size = 10),
        line = element_blank(),
        axis.text.x = element_text(hjust = 1, colour = "#2c3539", size = 10),
        axis.text.y = element_text(colour = "#2c3539", size = 10),
        axis.line.y = element_line(colour = "#2c3539"),
        axis.line.x = element_line(colour = "#2c3539"),
        axis.title = element_text(colour = "#2c3539", size = 10))

multiplot(plot_SalePrice_7, plot_SalePrice_8, cols = 2)

highQual_lowSalePrice <- train %>%
  filter(SalePrice < 300000.0 & GrLivArea > 4000 & OverallQual == "10")

bin.number <- nclass.FD(train$GrLivArea)
bin.width.GrLivArea <- max(train$GrLivArea)/bin.number
ggplot(train, aes(GrLivArea)) +
  geom_histogram(binwidth = bin.width.GrLivArea, colour = "#00ced1", fill = "#2c3539") +
  ylab("Count") +
  theme(rect = element_blank(),
        text = element_text(colour = "#2c3539", size = 12),
        line = element_blank(),
        axis.text.x = element_text(hjust = 1, colour = "#2c3539", size = 12),
        axis.text.y = element_text(colour = "#2c3539", size = 12),
        axis.line.y = element_line(colour = "#2c3539"),
        axis.line.x = element_line(colour = "#2c3539"),
        axis.title = element_text(colour = "#2c3539", size = 12))

ggplot(train, aes(GrLivArea, SalePrice)) +
  geom_point(colour = "#2c3539") +
  stat_smooth(method = "loess", colour = "#DC143C", size = 2, se = F) +
  ylab("Log of Sale Price") +
  xlab("Log of Ground Living Area") +
  ggtitle("Above Ground (x) ~ Sale Price (y)") +
  theme(rect = element_blank(),
        text = element_text(colour = "#2c3539", size = 12),
        line = element_blank(),
        axis.text.x = element_text(hjust = 1, colour = "#2c3539", size = 12),
        axis.text.y = element_text(colour = "#2c3539", size = 12),
        axis.line.y = element_line(colour = "#2c3539"),
        axis.line.x = element_line(colour = "#2c3539"),
        axis.title = element_text(colour = "#2c3539", size = 12))

ggplot(train, aes(group = OverallCond, OverallCond, SalePrice)) +
  geom_boxplot(colour = "#2c3539", fill = "#00ced1") +
  scale_y_continuous(labels = scales::dollar) +
  ylab("Sale Price") +
  xlab("Overall Condition") +
  ggtitle("Boxplot of SalePrice across each OverallCond rating") +
  theme(rect = element_blank(),
        text = element_text(colour = "#2c3539", size = 12),
        line = element_blank(),
        axis.text.x = element_text(hjust = 1, colour = "#2c3539", size = 12),
        axis.text.y = element_text(hjust = 1, colour = "#2c3539", size = 12),
        axis.line.x = element_line(colour = "#2c3539"),
        axis.line.y = element_line(colour = "#2c3539"),
        axis.title = element_text(colour = "#2c3539", size = 12))

filtered_train <- train %>%
  select(SalePrice, GrLivArea) %>%
  mutate(SalePrice_z = (SalePrice - mean(SalePrice))/sd(SalePrice),
         GrLivArea_z = (GrLivArea - mean(GrLivArea))/sd(GrLivArea)) %>%
  filter(SalePrice_z >= -3.0 & SalePrice_z <= 3.0,
         GrLivArea_z >= -3.0 & GrLivArea_z <= 3.0)

ggplot(filtered_train, aes(log10(GrLivArea), log10(SalePrice))) +
  geom_point(colour = "#2c3539") +
  stat_smooth(method = "lm", colour = "#00ced1", size = 2, se = F) +
  ylab("Log of Sale Price") +
  xlab("Log of Ground Living Area") +
  ggtitle("Above Ground (x) ~ Sale Price (y)") +
  theme(rect = element_blank(),
        text = element_text(colour = "#2c3539", size = 12),
        line = element_blank(),
        axis.text.x = element_text(hjust = 1, colour = "#2c3539", size = 12),
        axis.text.y = element_text(colour = "#2c3539", size = 12),
        axis.line.y = element_line(colour = "#2c3539"),
        axis.line.x = element_line(colour = "#2c3539"),
        axis.title = element_text(colour = "#2c3539", size = 12))

neighborhood_bar <- ggplot(train, aes(group = Neighborhood, Neighborhood, SalePrice)) +
  geom_boxplot(fill = "#002147", colour = "#a3c1ad") +
  ylab("Sale Price") +
  xlab("Neighborhood") +
  ggtitle("Neighborhood Sale Price Distribution") +
  scale_y_continuous(labels = scales::dollar) +
  theme(rect = element_blank(),
        text = element_text(colour = "#2c3539", size = 12),
        line = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1, colour = "#2c3539", size = 12),
        axis.text.y = element_text(colour = "#2c3539", size = 12),
        axis.line.y = element_line(colour = "#2c3539", size = 1.1),
        axis.line.x = element_line(colour = "#2c3539", size = 1.1),
        axis.title = element_text(colour = "#2c3539", size = 12))

neighborhood_boxplot <- ggplot(train, aes(Neighborhood)) +
  geom_bar(fill = "#a3c1ad", colour = "#002147") +
  ylab("count") +
  ggtitle("Neighborhood") +
  theme(rect = element_blank(),
        text = element_text(colour = "#2c3539", size = 12),
        line = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1, colour = "#2c3539", size = 12),
        axis.text.y = element_text(colour = "#2c3539", size = 12),
        axis.line.y = element_line(colour = "#2c3539", size = .7),
        axis.line.x = element_line(colour = "#2c3539", size = .7),
        axis.title = element_text(colour = "#2c3539", size = 12))

grid.arrange(neighborhood_bar, neighborhood_boxplot, ncol = 1)
#------------------------------------#
#----------- Variables --------------#
#------------------------------------#

########### Neighborhood ##############
ggplot(train, aes(reorder(Neighborhood, SalePrice, FUN = median), SalePrice)) +
  geom_boxplot(col = "#a3c1ad", fill = "#002147") +
  scale_y_continuous(labels = scales::dollar) +
  ylab("Sale Price") +
  xlab("Neighborhood") +
  theme(rect = element_blank(),
        text = element_text(colour = "#2c3539", size = 12),
        line = element_blank(),
        axis.text.x = element_text(angle = 35, hjust = 1, colour = "#2c3539", size = 12),
        axis.text.y = element_text(colour = "#2c3539", size = 12),
        axis.line.y = element_line(colour = "#2c3539", size = .7),
        axis.line.x = element_line(colour = "#2c3539", size = .7),
        axis.title = element_text(colour = "#2c3539", size = 12))



head(train$Neighb_Value)

ggplot(train, aes(Neighb_Value)) +
  geom_bar(fill = "002147", col = "2c3539") +
  ylab("Count") +
  theme_bw(base_size = 12)

bin.number <- nclass.FD(neighbs$Mean_SalePrice)
bin.width.Median <- max(neighbs$Mean_SalePrice)/bin.number
ggplot(neighbs, aes(Mean_SalePrice)) +
  geom_histogram(binwidth = bin.width.Median)

neighb_plot_1 <- ggplot(neighbs, aes(Neighborhood_Values)) +
  geom_bar(fill = "#002147", col = "#2c3539") +
  ylab("Count") +
  theme_bw(base_size = 12)
neighb_plot_2 <- ggplot(neighbs, aes(Group = Neighborhood_Values, Neighborhood_Values, Mean_SalePrice)) +
  geom_boxplot(fill = "#002147", col = "#2c3539") +
  ylab("Mean Neighborhood Sale Price") +
  theme_bw(base_size = 12)
grid.arrange(neighb_plot_1, neighb_plot_2, ncol = 1)

DecadeBuilt = cut(YearBuilt, breaks = 14, labels = c(1:14))
########### SalePrice ##############
summary(train.NA.rm$SalePrice)
str(train.NA.rm$SalePrice)
mean(train.NA.rm$SalePrice)
sd(train.NA.rm$SalePrice)

# Bin size - SalePrice
bin.number <- nclass.FD(train.NA.rm$SalePrice)
bin.width.SalePrice<- max(train.NA.rm$SalePrice)/bin.number

# Histogram
plot_SalePrice <- ggplot(train.NA.rm, aes(SalePrice)) +
  geom_histogram(binwidth = bin.width.SalePrice) +
  scale_x_continuous(labels = scales::dollar) +
  ylab("Count") +
  theme_bw(base_size = 12)

bin.number <- nclass.FD(train.NA.rm$SalePrice_Z)
bin.width.SalePrice_Z <- max(train.NA.rm$SalePrice_Z)/bin.number

# Z-Score
ggplot(train.NA.rm, aes(SalePrice_Z)) +
  geom_histogram(binwidth = bin.width.SalePrice_Z) +
  ylab("Count") +
  theme_bw(base_size = 12)

########### MSSubClass ##############  *****Keep for Models*****
train$MSSubClass <- as.factor(train$MSSubClass)

ggplot(train, aes(MSSubClass)) +
  geom_bar()

ggplot(train, aes(group = MSSubClass, MSSubClass, SalePrice)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::dollar) +
  facet_wrap(~MSZoning)

mean.price <- train %>%
  group_by(MSSubClass) %>%
  summarise(mean = mean(SalePrice))

ggplot(mean.price, aes(MSSubClass, mean)) +
  geom_col()

########### MSZoning ##############
train$MSZoning <- as.factor(train$MSZoning)

ggplot(train, aes(MSZoning)) +
  geom_bar()

ggplot(train, aes(group = MSZoning, MSZoning, SalePrice)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::dollar)

########### LotFrontage ############## *****Keep for Models*****
# Make sure to filter() 2.5:3 SD's

ggplot(train, aes(LotFrontage)) +
  geom_histogram()

ggplot(train, aes(LotFrontage, SalePrice)) +
  geom_point()

# Impute better values!
mean(train$LotFrontage)
train$LotFrontage[is.na(train$LotFrontage)] <- 57.80068

########### LotArea ############## *****Keep for Models*****
# Make sure to filter() 2.5:3 SD's

# Bin size- LotArea
bin.number <- nclass.FD(train$LotArea)
bin.width.LotArea <- max(train$LotArea)/bin.number

ggplot(train, aes(LotArea)) +
  geom_histogram(binwidth = bin.width.LotArea) +
  scale_x_continuous(labels = scales::comma) +
  ylab("Count") +
  theme_bw(base_size = 15) +
  theme(axis.text.x = element_text(angle = 75, hjust = 1))

ggplot(train, aes(LotArea, SalePrice)) +
  geom_point()


# LotArea Z Scores
Zscore_LotArea <- train.NA.rm %>%
  select(SalePrice, LotArea) %>%
  mutate(LotArea_ZScore = round((LotArea - mean(LotArea))/sd(LotArea), 2))

bin.number <- nclass.FD(Zscore_LotArea$LotArea_ZScore)
bin.width.LotArea_ZScore <- max(Zscore_LotArea$LotArea_ZScore)/bin.number

ggplot(Zscore_LotArea, aes(LotArea_ZScore))  +
  geom_histogram(binwidth = bin.width.LotArea_ZScore) +
  geom_vline(xintercept = 3, col = "red") +
  scale_x_continuous(labels = scales::comma) +
  ylab("Count") +
  theme_bw(base_size = 15) +
  theme(axis.text.x = element_text(angle = 75, hjust = 1))

summary(Zscore_LotArea)
sd(Zscore_LotArea$LotArea_ZScore)

Lot_Area_Adj <- Zscore_LotArea %>%
  filter(LotArea_ZScore <= 3.0 & LotArea_ZScore >= -3.0 )

ggplot(Lot_Area_Adj, aes(LotArea, SalePrice)) +
  geom_jitter(size = 2.5, alpha = .85, colour = "#003366", height = .5, width = .5) +
  stat_smooth(method = "lm", colour = "#00ced1", se = F) +
  scale_x_continuous(labels = scales::comma) +
  scale_y_continuous(labels = scales::dollar) +
  ylab("Count") +
  theme_bw(base_size = 15) +
  theme(axis.text.x = element_text(angle = 75, hjust = 1))

########### Street ##############
ggplot(train, aes(Street)) +
  geom_bar()

ggplot(train, aes(group = Street, Street, SalePrice)) +
  geom_boxplot()

########### Alley ##############
ggplot(train, aes(Alley)) +
  geom_bar()

ggplot(train, aes(group = Alley, Alley, SalePrice)) +
  geom_boxplot()

########### LotShape ############## *****Keep for Models*****
ggplot(train, aes(LotShape)) +
  geom_bar()

ggplot(train, aes(group = LotShape, LotShape, SalePrice)) +
  geom_boxplot()

########### LandContour ############## *****Keep for Models*****
ggplot(train, aes(LandContour)) +
  geom_bar()

ggplot(train, aes(group = LandContour, LandContour, SalePrice)) +
  geom_boxplot()

########### Utilities ##############
ggplot(train, aes(Utilities)) +
  geom_bar()

########### LotConfig ############## *****Keep for Models*****
ggplot(train, aes(LotConfig)) +
  geom_bar()

ggplot(train, aes(group = LotConfig, LotConfig, SalePrice)) +
  geom_boxplot()


########### LandSlope ##############
ggplot(train, aes(LandSlope)) +
  geom_bar()

ggplot(train, aes(group = LandSlope, LandSlope, SalePrice)) +
  geom_boxplot()

########### Neighborhood ############## *****Keep for Models*****
ggplot(train, aes(Neighborhood)) +
  geom_bar()

ggplot(train, aes(group = Neighborhood, Neighborhood, SalePrice)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::dollar)

########### Condition1 ############## *****Keep for Models*****
ggplot(train, aes(Condition1)) +
  geom_bar()

ggplot(train, aes(group = Condition1, Condition1, SalePrice)) +
  geom_boxplot()

########### Condition1 ##############
ggplot(train, aes(Condition2)) +
  geom_bar()

########### BldgType ############## *****Keep for Models*****
ggplot(train, aes(BldgType)) +
  geom_bar()

ggplot(train, aes(group = BldgType, BldgType, SalePrice)) +
  geom_boxplot()

########### HouseStyle ############## *****Keep for Models*****
ggplot(train, aes(HouseStyle)) +
  geom_bar()

ggplot(train, aes(group = HouseStyle, HouseStyle, SalePrice)) +
  geom_boxplot()

########### OverallQual ############## *****Keep for Models*****
train$OverallQual <- as.numeric(train$OverallQual)
ggplot(train, aes(OverallQual)) +
  geom_bar()

ggplot(train, aes(group = OverallQual, OverallQual, SalePrice)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::dollar)

########### OverallCond ############## *****Keep for Models*****
ggplot(train, aes(OverallCond)) +
  geom_bar()

ggplot(train, aes(group = OverallCond, OverallCond, SalePrice)) +
  geom_boxplot()

########### YearBuilt ############## *****Keep for Models*****
ggplot(train, aes(YearBuilt)) +
  geom_bar()

ggplot(train, aes(group = YearBuilt, YearBuilt, SalePrice)) +
  geom_boxplot(colour = "#2c3539", fill = "#00ced1") +
  scale_y_continuous(labels = scales::dollar)

########### YearRemodAdd ############## *****Keep for Models*****
ggplot(train, aes(YearRemodAdd)) +
  geom_bar()

ggplot(train, aes(group = YearRemodAdd, YearRemodAdd, SalePrice)) +
  geom_boxplot()

########### RoofStyle ############## *****Keep for Models*****
ggplot(train, aes(RoofStyle)) +
  geom_bar()

ggplot(train, aes(group = RoofStyle, RoofStyle, SalePrice)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::dollar)

########### RoofMatl ##############
ggplot(train, aes(RoofMatl)) +
  geom_bar()

########### Exterior1st ############## *****Keep for Models*****
ggplot(train, aes(Exterior1st)) +
  geom_bar()

ggplot(train, aes(group = Exterior1st, Exterior1st, SalePrice)) +
  geom_boxplot(colour = "#2c3539", fill = "#00ced1") +
  scale_y_continuous(labels = scales::dollar)

########### Exterior2nd ############## *****Keep for Models*****
ggplot(train, aes(Exterior2nd)) +
  geom_bar()

ggplot(train, aes(group = Exterior2nd, Exterior2nd, SalePrice)) +
  geom_boxplot(colour = "#2c3539", fill = "#00ced1") +
  scale_y_continuous(labels = scales::dollar)

########### MasVnrType ############## *****Keep for Models*****
ggplot(train, aes(MasVnrType)) +
  geom_bar()

ggplot(train, aes(group = MasVnrType, MasVnrType, SalePrice)) +
  geom_boxplot(colour = "#2c3539", fill = "#00ced1") +
  scale_y_continuous(labels = scales::dollar)

########### MasVnrArea ##############



ggplot(train.NA.rm, aes(group = Fireplaces, Fireplaces, SalePrice)) +
  geom_boxplot()
ggplot(train.NA.rm, aes(group = GarageCars, GarageCars, SalePrice)) +
  geom_boxplot()
ggplot(train.NA.rm, aes(group = YrSold, YrSold, SalePrice)) +
  geom_boxplot()

# Decade Built
plot_DecadeBuilt <- ggplot(train.NA.rm, aes(DecadeBuilt)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12)

ggplot(train.NA.rm, aes(group = DecadeBuilt, DecadeBuilt, SalePrice)) +
  geom_boxplot(colour = "#2c3539", fill = "#C0C0C0") +
  scale_y_continuous(labels = scales::dollar) +
  ylab("Sale Price") +
  xlab("Decade Built") +
  theme_bw(base_size = 12)

ggplot(train.NA.rm, aes(DecadeBuilt, Bath)) +
  geom_count()

ggplot(train.NA.rm, aes(DecadeBuilt, factor(OverallQual))) +
  geom_count()

# Bath
BsmtFullBath
BsmtHalfBath
FullBath
HalfBath

str(train.NA.rm[,c("BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath")])

bin.number <- nclass.FD(train.NA.rm$Bath)
bin.width.Bath <- max(train.NA.rm$Bath)/bin.number
plot_Bath <- ggplot(train.NA.rm, aes(Bath)) +
  geom_histogram(binwidth = bin.width.Bath, fill = "#C0C0C0", colour= "#2c3539") +
  scale_x_continuous(breaks = c(1, 2, 3, 4, 5, 6)) +
  scale_y_continuous(breaks = c(100, 200, 300, 400, 500)) +
  ylab("Count") +
  theme_bw(base_size = 12)

plot_Bath_boxplot <- ggplot(train.NA.rm, aes(group = Bath, Bath, SalePrice)) +
  geom_boxplot(colour = "#2c3539", fill = "#C0C0C0") +
  ylab("Sale Price") +
  xlab("Number of Bathrooms") +
  theme_bw(base_size = 12)

grid.arrange(plot_Bath, plot_Bath_boxplot, ncol = 1)

# Bin size - OverallQual
bin.number <- nclass.FD(train.NA.rm$OverallQual)
bin.width.OverallQual <- max(train.NA.rm$OverallQual)/bin.number

# Bin size - OverallCond
bin.number <- nclass.FD(train.NA.rm$OverallCond)
bin.width.OverallCond <- max(train.NA.rm$OverallCond)/bin.number

#Bin Size - BsmtFinSF1
bin.number <- nclass.FD(train.NA.rm$BsmtFinSF1)
bin.width.BsmtFinSF1 <- max(train.NA.rm$BsmtFinSF1)/bin.number

# Bin size = BsmtUnfSF
bin.number <- nclass.FD(train.NA.rm$BsmtUnfSF)
bin.width.BsmtUnfSF <- max(train.NA.rm$BsmtUnfSF)/bin.number

# Bin Size - FirstFloorSF
bin.number <- nclass.FD(train.NA.rm$FirstFloorSF)
bin.width.FirstFloorSF <- max(train.NA.rm$FirstFloorSF)/bin.number

# Bin size - SecondFloorSF
bin.number <- nclass.FD(train.NA.rm$SecondFloorSF)
bin.width.SecondFloorSF <- max(train.NA.rm$SecondFloorSF)/bin.number

# Bin Size - LowQualFinSF
bin.number <- nclass.FD(train.NA.rm$LowQualFinSF)
bin.width.LowQualFinSF <- max(train.NA.rm$LowQualFinSF)/bin.number

# Bin size - GrLivArea 
bin.number <- nclass.FD(train.NA.rm$GrLivArea)
bin.width.GrLivArea <- max(train.NA.rm$GrLivArea)/bin.number

# Bin size - GarageArea
bin.number <- nclass.FD(train.NA.rm$GarageArea)
bin.width.GarageArea <- max(train.NA.rm$GarageArea)/bin.number

# Bin size - Wood Deck
bin.number <- nclass.FD(train.NA.rm$WoodDeckSF)
bin.width.WoodDeckSF <- max(train.NA.rm$WoodDeckSF)/bin.number

# Bin size - Open Porch
bin.number <- nclass.FD(train.NA.rm$OpenPorchSF)
bin.width.OpenPorchSF <- max(train.NA.rm$OpenPorchSF)/bin.number

# Bin size - Enclosed Porch
bin.number <- nclass.FD(train.NA.rm$EnclosedPorch)
bin.width.EnclosedPorch <- max(train.NA.rm$EnclosedPorch)/bin.number

# Bin size - 3 Season Porch
bin.number <- nclass.FD(train.NA.rm$ThreeSeasonPorch)
bin.width.ThreeSeasonPorch <- max(train.NA.rm$ThreeSeasonPorch)

# Bin size - Pool Area
bin.number <- nclass.FD(train.NA.rm$PoolArea)
bin.width.PoolArea <- max(train.NA.rm$PoolArea)

# Univariate Plots

plot_MSSubClass <- ggplot(train.NA.rm, aes(MSSubClass)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_MSZoning <- ggplot(train.NA.rm, aes(MSZoning)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12)

plot_Street <- ggplot(train.NA.rm, aes(Street)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_LotShape <- ggplot(train.NA.rm, aes(LotShape)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_LandContour <- ggplot(train.NA.rm, aes(LandContour)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_Utilities <- ggplot(train.NA.rm, aes(Utilities)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_LotConfig <- ggplot(train.NA.rm, aes(LotConfig)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_LandSlope <- ggplot(train.NA.rm, aes(LandSlope)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_Neighborhood <- ggplot(train.NA.rm, aes(Neighborhood)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12) +
  theme(axis.text.x = element_text(angle = 75, hjust = 1))
plot_Condition1 <- ggplot(train.NA.rm, aes(Condition1)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12) +
  theme(axis.text.x = element_text(angle = 75, hjust = 1))
plot_Condition2 <- ggplot(train.NA.rm, aes(Condition2)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12) +
  theme(axis.text.x = element_text(angle = 75, hjust = 1))
plot_BldgType <- ggplot(train.NA.rm, aes(BldgType)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_HouseStyle <- ggplot(train.NA.rm, aes(HouseStyle)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12) +
  theme(axis.text.x = element_text(angle = 75, hjust = 1))
plot_OverallQual <- ggplot(train.NA.rm, aes(OverallQual)) +
  geom_histogram(binwidth = bin.width.OverallQual) +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_OverallCond <- ggplot(train.NA.rm, aes(OverallCond)) +
  geom_histogram(binwidth = bin.width.OverallCond) +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_YearBuilt <- ggplot(train.NA.rm, aes(YearBuilt)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_YearRemodAdd <- ggplot(train.NA.rm, aes(YearRemodAdd)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_RoofStyle <- ggplot(train.NA.rm, aes(RoofStyle)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12) +
  theme(axis.text.x = element_text(angle = 75, hjust = 1))
plot_RoofMatl <- ggplot(train.NA.rm, aes(RoofMatl)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12) +
  theme(axis.text.x = element_text(angle = 75, hjust = 1))
plot_Exterior1st <- ggplot(train.NA.rm, aes(Exterior1st)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12) +
  theme(axis.text.x = element_text(angle = 75, hjust = 1))
plot_Exterior2nd <- ggplot(train.NA.rm, aes(Exterior2nd)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12) +
  theme(axis.text.x = element_text(angle = 75, hjust = 1))
plot_ExterQual <- ggplot(train.NA.rm, aes(ExterQual)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_ExterCond <- ggplot(train.NA.rm, aes(ExterCond)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_Foundation <- ggplot(train.NA.rm, aes(Foundation)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12) +
  theme(axis.text.x = element_text(angle = 75, hjust = 1))
plot_BsmtFinSF1 <- ggplot(train.NA.rm, aes(BsmtFinSF1)) +
  geom_histogram(binwidth = bin.width.BsmtFinSF1) +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_BsmtUnfSF <- ggplot(train.NA.rm, aes(BsmtUnfSF)) +
  geom_histogram(binwidth = bin.width.BsmtUnfSF) +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_Heating <- ggplot(train.NA.rm, aes(Heating)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12) +
  theme(axis.text.x = element_text(angle = 75, hjust = 1))
plot_HeatingQC <- ggplot(train.NA.rm, aes(HeatingQC)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12) +
  theme(axis.text.x = element_text(angle = 75, hjust = 1))
plot_CentralAir <- ggplot(train.NA.rm, aes(CentralAir)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12) +
  theme(axis.text.x = element_text(angle = 75, hjust = 1))
plot_FirstFloorSF <- ggplot(train.NA.rm, aes(FirstFloorSF)) +
  geom_histogram(binwidth = bin.width.FirstFloorSF) +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_SecondFloorSF <- ggplot(train.NA.rm, aes(SecondFloorSF)) +
  geom_histogram(binwidth = bin.width.SecondFloorSF) +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_LowQualFinSF <- ggplot(train.NA.rm, aes(LowQualFinSF)) +
  geom_histogram(binwidth = bin.width.LowQualFinSF) +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_GrLivArea <- ggplot(train.NA.rm, aes(GrLivArea)) +
  geom_histogram(binwidth = bin.width.GrLivArea) +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_BsmtFullBath <- ggplot(train.NA.rm, aes(BsmtFullBath)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_BsmtHalfBath <- ggplot(train.NA.rm, aes(BsmtHalfBath)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_FullBath <- ggplot(train.NA.rm, aes(FullBath)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_HalfBath <- ggplot(train.NA.rm, aes(HalfBath)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_Bedroom <- ggplot(train.NA.rm, aes(Bedroom)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_Kitchen <- ggplot(train.NA.rm, aes(Kitchen)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_KitchenQual <- ggplot(train.NA.rm, aes(KitchenQual)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_TotRmsAbvGrd <- ggplot(train.NA.rm, aes(TotRmsAbvGrd)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_Functional <- ggplot(train.NA.rm, aes(Functional)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12) +
  theme(axis.text.x = element_text(angle = 75, hjust = 1))
plot_Fireplaces <- ggplot(train.NA.rm, aes(Fireplaces)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_GarageCars <- ggplot(train.NA.rm, aes(GarageCars)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_GarageArea <- ggplot(train.NA.rm, aes(GarageArea)) +
  geom_histogram(binwidth = bin.width.GarageArea) +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_PavedDrive <- ggplot(train.NA.rm, aes(PavedDrive)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_WoodDeckSF <- ggplot(train.NA.rm, aes(WoodDeckSF)) +
  geom_histogram(binwidth = bin.width.WoodDeckSF) +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_OpenPorchSF <- ggplot(train.NA.rm, aes(OpenPorchSF)) +
  geom_histogram(binwidth = bin.width.OpenPorchSF) +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_EnclosedPorch <- ggplot(train.NA.rm, aes(EnclosedPorch)) +
  geom_histogram(binwidth = bin.width.EnclosedPorch) +
  ylab("Count") +
  theme_bw(base_size = 12)

#----------Un-Useful--------
plot_ThreeSeasonPorch <- ggplot(train.NA.rm, aes(ThreeSeasonPorch)) +
  geom_histogram(binwidth = bin.width.ThreeSeasonPorch) +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_PoolArea <- ggplot(train.NA.rm, aes(PoolArea)) +
  geom_histogram(binwidth = bin.width.PoolArea) +
  ylab("Count") +
  theme_bw(base_size = 12)
#---------------------------

plot_MoSold <- ggplot(train.NA.rm, aes(MoSold)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_YrSold <- ggplot(train.NA.rm, aes(YrSold)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12)
plot_SaleType <- ggplot(train.NA.rm, aes(SaleType)) +
  geom_bar() +
  ylab("Count") +
  theme_bw(base_size = 12)

bin.number <- nclass.FD(train.NA.rm$SalePrice)
bin.width.SalePrice <- max(train.NA.rm$SalePrice)/bin.number
plot_SalePrice <- ggplot(train.NA.rm, aes(SalePrice)) +
  geom_histogram(binwidth = bin.width.SalePrice) +
  scale_x_continuous(labels = scales::dollar) +
  ylab("Count") +
  theme_bw(base_size = 12)

plot_SalePrice_Density <- ggplot(train.NA.rm, aes(SalePrice)) +
  geom_density(kernel = "gaussian") +
  scale_x_continuous(labels = scales::dollar) +
  ylab("Count") +
  theme_bw(base_size = 12)

train.NA.rm <- train.NA.rm %>%
  mutate(SalePrice_Z = (SalePrice - mean(SalePrice))/sd(SalePrice))
bin.number <- nclass.FD(train.NA.rm$SalePrice_Z)
bin.width.SalePrice_Z <- max(train.NA.rm$SalePrice_Z)/bin.number
plot_SalePrice_Z <- ggplot(train.NA.rm, aes(SalePrice_Z)) +
  geom_histogram(binwidth = bin.width.SalePrice_Z) +
  ylab("Count") +
  theme_bw(base_size = 12)

bin.number <- nclass.FD(train.NA.rm$LotArea)
bin.width.LotArea <- max(train.NA.rm$LotArea)/bin.number
plot_LotArea <- ggplot(train.NA.rm, aes(LotArea)) +
  geom_histogram(binwidth = bin.width.LotArea) +
  scale_x_continuous(labels = scales::comma) +
  ylab("Count") +
  theme_bw(base_size = 12)

all_plot <- grid.arrange(plot_SalePrice,
             plot_MSSubClass,
             plot_MSZoning,
             plot_LotArea, 
             plot_Street,
             plot_LotShape,
             plot_LandContour,
             plot_Utilities,
             plot_LotConfig,
             plot_LandSlope,
             plot_Neighborhood,
             plot_Condition1,
             plot_Condition2,
             plot_BldgType,
             plot_HouseStyle,
             plot_OverallQual, 
             plot_OverallCond,
             plot_YearBuilt,
             plot_DecadeBuilt,
             plot_YearRemodAdd,
             plot_RoofStyle,
             plot_RoofMatl,
             plot_Exterior1st,
             plot_Exterior2nd,
             plot_ExterQual,
             plot_ExterCond,
             plot_Foundation,
             plot_BsmtFinSF1,
             plot_BsmtUnfSF,
             plot_Heating,
             plot_HeatingQC,
             plot_CentralAir,
             plot_FirstFloorSF,
             plot_SecondFloorSF,
             plot_LowQualFinSF,
             plot_GrLivArea,
             plot_BsmtFullBath,
             plot_BsmtHalfBath,
             plot_FullBath,
             plot_HalfBath,
             plot_Bath,
             plot_Bedroom,
             plot_Kitchen,
             plot_KitchenQual,
             plot_TotRmsAbvGrd,
             plot_Functional,
             plot_Fireplaces,
             plot_GarageCars,
             plot_GarageArea,
             plot_PavedDrive,
             plot_WoodDeckSF,
             plot_OpenPorchSF,
             plot_EnclosedPorch,
             plot_MoSold,
             plot_YrSold,
             plot_SaleType,
             ncol = 8)

# Scatterplots
ggplot(train.NA.rm, aes(YearBuilt, SalePrice)) +
  geom_point(size = 2.5, alpha = .85, colour = "#003366") +
  stat_smooth(method = "loess", colour = "#00ced1") +
  ylab("Sale Price") +
  xlab("Year Built") +
  scale_y_continuous(labels = scales::dollar) +
  theme_bw(base_size = 15)

cor(train.NA.rm$YearBuilt, train.NA.rm$SalePrice)
model_1_YearBuilt <- lm(SalePrice ~ YearBuilt, data = train.NA.rm)
summary(model_1_YearBuilt)

xy_plot_1 <- ggplot(train.NA.rm, aes(FirstFloorSF, SalePrice)) +
  geom_jitter(size = 2.5, alpha = .85, colour = "#003366", height = .5, width = .5) +
  stat_smooth(method = "lm", colour = "#00ced1") +
  ylab("Sale Price") +
  xlab("First Floor Square Feet") +
  scale_y_continuous(labels = scales::dollar) +
  theme_bw(base_size = 15)

cor(train.NA.rm$FirstFloorSF, train.NA.rm$SalePrice)
model_1_FirstFloorSF<- lm(SalePrice ~ FirstFloorSF, data = train.NA.rm)
summary(model_1_FirstFloorSF)

xy_plot_2 <- ggplot(train.NA.rm, aes(GrLivArea, SalePrice)) +
  geom_jitter(size = 2.5, alpha = .85, colour = "#003366", height = .5, width = .5) +
  stat_smooth(method = "lm", colour = "#00ced1") +
  ylab("Sale Price") +
  xlab("Above Ground Living Area") +
  scale_x_continuous(labels = scales::comma) +
  scale_y_continuous(labels = scales::dollar) +
  theme_bw(base_size = 12)

cor(train.NA.rm$GrLivArea, train.NA.rm$SalePrice)
model_1_GrLivArea <- lm(SalePrice ~ GrLivArea, data = train.NA.rm)
summary(model_1_GrLivArea)

ggplot(train.NA.rm, aes(GarageArea, SalePrice)) +
  geom_jitter(size = 2.5, alpha = .85, colour = "#003366", height = .5, width = .5) +
  stat_smooth(method = "lm", colour = "#00ced1") +
  ylab("Sale Price") +
  xlab("Garage Area") +
  scale_x_continuous(labels = scales::comma) +
  scale_y_continuous(labels = scales::comma) +
  theme_bw(base_size = 12)

cor(train.NA.rm$GarageArea, train.NA.rm$SalePrice)
model_1_GarageArea <- lm(SalePrice ~ GarageArea, data = train.NA.rm)
summary(model_1_GarageArea)

xy_plot_3 <- ggplot(train.NA.rm, aes(FirstFloorSF, GrLivArea)) +
  geom_jitter(size = 2.5, alpha = .85, colour = "#003366", height = .5, width = .5) +
  stat_smooth(method = "lm", colour = "#00ced1") +
  ylab("Above Ground Living Area") +
  xlab("First Floor Area") +
  scale_x_continuous(labels = scales::comma) +
  scale_y_continuous(labels = scales::comma) +
  theme_bw(base_size = 12)

cor(train.NA.rm$FirstFloorSF, train.NA.rm$GrLivArea)
model_1_FloorSpace <- lm(GrLivArea ~ FirstFloorSF, data = train.NA.rm)
summary(model_1_FloorSpace)

grid.arrange(xy_plot_1, xy_plot_2, xy_plot_3, ncol = 3)


# SANDBOX

summary(train.NA.rm[,c("BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath",
                       "Bedroom", "Kitchen")])

#------------------------------------------------------------------------------------------
ggplot(train.NA.rm, aes(SalePrice_Z)) +
  geom_histogram()

ggplot(train.NA.rm, aes(SalePrice)) +
  geom_histogram()
#------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------
mean.SalePrice.Exterior1st <- train.NA.rm %>%
  select(Exterior1st, SalePrice) %>%
  group_by(Exterior1st) %>%
  summarise(Mean_Sale_Price = mean(SalePrice)) %>%
  arrange(desc(Mean_Sale_Price))
mean.SalePrice.Exterior1st

ggplot(mean.SalePrice.Exterior1st, aes(Exterior1st, Mean_Sale_Price)) +
  geom_col() +
  ylab("Count") +
  scale_y_continuous(labels = scales::comma) +
  theme_bw(base_size = 12) +
  theme(axis.text.x = element_text(angle = 75, hjust = 1))

mean.SalePrice.Foundation <- train.NA.rm %>%
  select(Foundation, SalePrice) %>%
  group_by(Foundation) %>%
  summarise(Mean_Sale_Price = mean(SalePrice)) %>%
  arrange(desc(Mean_Sale_Price))
mean.SalePrice.Foundation

ggplot(mean.SalePrice.Foundation, aes(Foundation, Mean_Sale_Price)) +
  geom_col() +
  ylab("Count") +
  scale_y_continuous(labels = scales::comma) +
  theme_bw(base_size = 12) +
  theme(axis.text.x = element_text(angle = 75, hjust = 1))
#------------------------------------------------------------------------------------------



#-------------------------------------------------------------------------------------------
#                           Simple Regression on Continuous Variables
#-------------------------------------------------------------------------------------------


train.regression <- train %>%
  rename(FirstFloorSF = X1stFlrSF, SecondFloorSF = X2ndFlrSF,
         Bedroom = BedroomAbvGr, Kitchen = KitchenAbvGr,
         ThreeSeasonPorch = X3SsnPorch, Rooms = TotRmsAbvGrd) %>%
  select(SalePrice, LotArea, OverallQual, BsmtUnfSF, FirstFloorSF,
         GrLivArea, Rooms, GarageCars, GarageArea, MoSold) %>%
  mutate(SalePrice_Z = (SalePrice - mean(SalePrice))/sd(SalePrice),
         LotArea_Z = (LotArea - mean(LotArea))/sd(LotArea),
         BsmtUnfSF_Z = (BsmtUnfSF - mean(BsmtUnfSF))/sd(BsmtUnfSF),
         FirstFloorSF_Z = (FirstFloorSF - mean(FirstFloorSF))/sd(FirstFloorSF),
         GrLivArea_Z = (GrLivArea - mean(GrLivArea))/sd(GrLivArea),
         Rooms_Z = (Rooms - mean(Rooms))/sd(Rooms),
         GarageCars_Z = (GarageCars - mean(GarageCars))/sd(GarageCars),
         GarageArea_Z = (GarageArea - mean(GarageArea))/sd(GarageArea)) %>%
  filter(SalePrice_Z >= -3.0 & SalePrice_Z <= 3.0,
         LotArea_Z >= -3.0 & LotArea_Z <= 3.0,
         BsmtUnfSF_Z >= -3.0 & BsmtUnfSF_Z <= 3.0,
         FirstFloorSF_Z >= -3.0 & FirstFloorSF_Z <= 3.0,
         GrLivArea_Z >= -3.0 & GrLivArea_Z <= 3.0,
         Rooms_Z >= -3.0 & Rooms_Z <= 3.0,
         GarageCars_Z >= -3.0 & GarageCars_Z <= 3.0,
         GarageArea_Z >= -3.0 & GarageArea_Z <= 3.0) %>%
  select(-c(SalePrice_Z, LotArea_Z, BsmtUnfSF_Z, FirstFloorSF_Z, GrLivArea_Z,
            Rooms_Z, GarageCars_Z, GarageArea_Z))

mean(train.regression$SalePrice)

lm_SalePrice_Continuous <- lm(SalePrice ~ ., data = train.regression)
summary(lm_SalePrice_Continuous)

test.regression <- test %>%
  rename(FirstFloorSF = X1stFlrSF, SecondFloorSF = X2ndFlrSF,
         Bedroom = BedroomAbvGr, Kitchen = KitchenAbvGr,
         ThreeSeasonPorch = X3SsnPorch, Rooms = TotRmsAbvGrd) %>%
  select(Id, LotArea, OverallQual, BsmtUnfSF, FirstFloorSF,
         GrLivArea, Rooms, GarageCars, GarageArea, MoSold)

preds.regression <- predict(lm_SalePrice_Continuous, test.regression)
test.Id <- test.regression %>%
  select(Id)
Test.Submission.LM.Continuous <- cbind(test.Id, preds.regression)
Test.Submission.LM.Continuous <- Test.Submission.LM.Continuous %>%
  rename(SalePrice = preds.regression)

Test.Submission.LM.Continuous$SalePrice[is.na(Test.Submission.LM.Continuous$SalePrice)] <- mean(train.regression$SalePrice)

write.csv(Test.Submission.LM.Continuous, file = "Test.Submission.LM.Continuous.csv")

#-------------------------------------------------------------------------------------------
#                           GBM on Continuous Variables
#-------------------------------------------------------------------------------------------

train.regression <- train %>%
  rename(FirstFloorSF = X1stFlrSF, SecondFloorSF = X2ndFlrSF,
         Bedroom = BedroomAbvGr, Kitchen = KitchenAbvGr,
         ThreeSeasonPorch = X3SsnPorch, Rooms = TotRmsAbvGrd) %>%
  select(SalePrice, LotArea, OverallQual, BsmtUnfSF, FirstFloorSF,
         GrLivArea, Rooms, GarageCars, GarageArea, MoSold, YearBuilt) %>%
  mutate(SalePrice_Z = (SalePrice - mean(SalePrice))/sd(SalePrice),
         LotArea_Z = (LotArea - mean(LotArea))/sd(LotArea),
         BsmtUnfSF_Z = (BsmtUnfSF - mean(BsmtUnfSF))/sd(BsmtUnfSF),
         FirstFloorSF_Z = (FirstFloorSF - mean(FirstFloorSF))/sd(FirstFloorSF),
         GrLivArea_Z = (GrLivArea - mean(GrLivArea))/sd(GrLivArea),
         Rooms_Z = (Rooms - mean(Rooms))/sd(Rooms),
         GarageCars_Z = (GarageCars - mean(GarageCars))/sd(GarageCars),
         GarageArea_Z = (GarageArea - mean(GarageArea))/sd(GarageArea)) %>%
  filter(SalePrice_Z >= -3.0 & SalePrice_Z <= 3.0,
         LotArea_Z >= -3.0 & LotArea_Z <= 3.0,
         BsmtUnfSF_Z >= -3.0 & BsmtUnfSF_Z <= 3.0,
         FirstFloorSF_Z >= -3.0 & FirstFloorSF_Z <= 3.0,
         GrLivArea_Z >= -3.0 & GrLivArea_Z <= 3.0,
         Rooms_Z >= -3.0 & Rooms_Z <= 3.0,
         GarageCars_Z >= -3.0 & GarageCars_Z <= 3.0,
         GarageArea_Z >= -3.0 & GarageArea_Z <= 3.0) %>%
  select(-c(SalePrice_Z, LotArea_Z, BsmtUnfSF_Z, FirstFloorSF_Z, GrLivArea_Z,
            Rooms_Z, GarageCars_Z, GarageArea_Z))

train.regression$YearBuilt <- as.character(train.regression$YearBuilt)
train.regression$YearBuilt <- as.Date(train.regression$YearBuilt, "%Y")

bin.number <- nclass.FD(train.regression$YearBuilt)
bin.width.YearBuilt <- max(train.regression$YearBuilt)

ggplot(train.regression, aes(YearBuilt)) +
  geom_bar(colour = "#C0C0C0", fill = "#2c3539") +
  ylab("Count") +
  theme_bw(base_size = 12)

bin.number <- nclass.FD(train.regression$LotArea)
bin.width.LotArea_REG <- max(train.regression$LotArea)/bin.number

ggplot(train.regression, aes(LotArea)) +
  geom_histogram(binwidth = bin.width.LotArea_REG) +
  ylab("Count") +
  scale_x_continuous(labels = scales::comma) +
  theme_bw(base_size = 12)

ggplot(train.regression, aes(group = YearBuilt, YearBuilt, SalePrice)) +
  geom_boxplot(colour = "#2c3539", fill = "#003366") +
  scale_y_continuous(labels = scales::dollar) +
  scale_x_continuous(breaks = c(1872, 1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010)) +
  ylab("Sale Price") +
  ggtitle("House Prices - Ames, Iowa (1872-2010)") +
  theme(rect = element_blank(),
        text = element_text(colour = "#2c3539", size = 25),
        line = element_blank(),
        axis.text.x = element_text(hjust = 1, colour = "#2c3539"),
        axis.text.y = element_text(colour = "#2c3539"),
        axis.line.y = element_line(colour = "#2c3539"),
        axis.line.x = element_line(colour = "#2c3539"),
        axis.title = element_text(colour = "#2c3539"))

ggplot(train, aes(LotFrontage, SalePrice)) +
  geom_jitter(colour = "#2c3539", width = 1) +
  stat_smooth(method = "lm", col = "red") +
  facet_wrap(~Neighborhood) +
  scale_y_continuous(labels = scales::dollar) +
  ggtitle("House Price - Ames, Iowa (1972 - 2010)") +
  theme(rect = element_blank(),
        text = element_text(colour = "#2c3539", size = 25),
        line = element_blank(),
        axis.text.x = element_text(hjust = 1, colour = "#2c3539"),
        axis.text.y = element_text(colour = "#2c3539"),
        axis.line.y = element_line(colour = "#2c3539"),
        axis.line.x = element_line(colour = "#2c3539"),
        axis.title = element_text(colour = "#2c3539"))

ggplot(train.regression, aes(OverallQual, SalePrice)) +
  geom_jitter(colour = "#2c3539", width = 1) +
  stat_smooth(method = "lm", col = "red") +
  scale_y_continuous(labels = scales::dollar) +
  ggtitle("House Prices - Ames, Iowa (1872 - 2010)") +
  theme(rect = element_blank(),
        text = element_text(colour = "#2c3539", size = 25),
        line = element_blank(),
        axis.text.x = element_text(hjust = 1, colour = "#2c3539"),
        axis.text.y = element_text(colour = "#2c3539"),
        axis.line.y = element_line(colour = "#2c3539"),
        axis.line.x = element_line(colour = "#2c3539"),
        axis.title = element_text(colour = "#2c3539"))

# Test, Train Split
set.seed(222333)
splitIndex <- createDataPartition(train.regression$SalePrice,
                                  p = .6,
                                  list = F,
                                  times = 1)

gbm.train.1 <- train.regression

gbm.label <- train.regression$SalePrice
gbm.train.1$SalePrice <- NULL

# 10-fold Cross Validation
cv.10.folds <- createMultiFolds(gbm.label, k = 10, times = 10)

metric <- "RMSE"

objControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10,
                           index = cv.10.folds)

# Parameter Grid Search
gbm.grid.1 <- expand.grid(interaction.depth = c(1, 2, 4, 8),
                          shrinkage = c(0.1, 0.01, 0.001),
                          n.trees = c(100, 200, 400, 800),
                          n.minobsinnode = c(1, 2, 4, 8))

# Best Tune
gbm.grid.1.best <- expand.grid(interaction.depth = 8,
                               shrinkage = 0.01,
                               n.trees = 800,
                               n.minobsinnode = 1)

# Model Training
c1 <- makeCluster(4, type = "SOCK")
registerDoSNOW(c1)

set.seed(123456)
gbm.1.cv.1 <- train(x = gbm.train.1, y = gbm.label,
                    distribution = "gaussian",
                    method = "gbm",
                    trControl = objControl,
                    tuneGrid = gbm.grid.1.best,
                    metric = metric,
                    preProcess = c("center", "scale"))

stopCluster(c1)

summary(gbm.1.cv.1)
plot(varImp(gbm.1.cv.1))
gbm.1.cv.1

# Submission

test.regression <- test %>%
  rename(FirstFloorSF = X1stFlrSF, SecondFloorSF = X2ndFlrSF,
         Bedroom = BedroomAbvGr, Kitchen = KitchenAbvGr,
         ThreeSeasonPorch = X3SsnPorch, Rooms = TotRmsAbvGrd) %>%
  select(Id, LotArea, OverallQual, BsmtUnfSF, FirstFloorSF,
         GrLivArea, Rooms, GarageCars, GarageArea, MoSold, YearBuilt)

preds.GBM <- predict(gbm.1.cv.1, test.regression)

test.Id <- test.regression %>%
  select(Id)

Test.Submission.GBM.Continuous <- cbind(test.Id, preds.GBM)
Test.Submission.GBM.Continuous <- Test.Submission.GBM.Continuous %>%
  rename(SalePrice = preds.GBM)

write.csv(Test.Submission.GBM.Continuous, file = "Test.Submission.GBM.Continuous.csv")

bin.number <- nclass.FD(train.regression$SalePrice)
bin.width.ActualSalePrice <- max(train.regression$SalePrice)/bin.number
bin.number <- nclass.FD(Test.Submission.GBM.Continuous$SalePrice)
bin.width.GBMSalePrice <- max(Test.Submission.GBM.Continuous$SalePrice)/bin.number

preds.hist <- ggplot(Test.Submission.GBM.Continuous, aes(SalePrice)) +
  geom_histogram(binwidth = bin.width.GBMSalePrice, colour = "#FFFFFF") +
  ylab("Count") +
  xlab("Predicted Test Sale Price") +
  scale_x_continuous(labels = scales::dollar) +
  theme_bw(base_size = 12)
actual.hist <- ggplot(train.regression, aes(SalePrice)) +
  geom_histogram(binwidth = bin.width.ActualSalePrice, colour = "#FFFFFF") +
  ylab("Count") +
  xlab("Training Sale Price") +
  scale_x_continuous(labels = scales::dollar) +
  theme_bw(base_size = 12)
grid.arrange(preds.hist, actual.hist, ncol = 2)

#-------------------------------------------------------------------------------------------
#                           GBM on Continuous Variables with Transformations
#-------------------------------------------------------------------------------------------

train.regression <- train %>%
  rename(FirstFloorSF = X1stFlrSF, SecondFloorSF = X2ndFlrSF,
         Bedroom = BedroomAbvGr, Kitchen = KitchenAbvGr,
         ThreeSeasonPorch = X3SsnPorch, Rooms = TotRmsAbvGrd) %>%
  select(SalePrice, LotArea, OverallQual, BsmtUnfSF, FirstFloorSF,
         GrLivArea, Rooms, GarageCars, GarageArea, YearBuilt) %>%
  mutate(OverallQual_Z = (OverallQual - mean(OverallQual))/sd(OverallQual),
         LotArea_Z = (LotArea - mean(LotArea))/sd(LotArea),
         BsmtUnfSF_Z = (BsmtUnfSF - mean(BsmtUnfSF))/sd(BsmtUnfSF),
         FirstFloorSF_Z = (FirstFloorSF - mean(FirstFloorSF))/sd(FirstFloorSF),
         GrLivArea_Z = (GrLivArea - mean(GrLivArea))/sd(GrLivArea),
         Rooms_Z = (Rooms - mean(Rooms))/sd(Rooms),
         GarageCars_Z = (GarageCars - mean(GarageCars))/sd(GarageCars),
         GarageArea_Z = (GarageArea - mean(GarageArea))/sd(GarageArea)) %>%
  mutate(OverallQual = log(OverallQual_Z^2),
         LotArea = log(LotArea_Z^2),
         BsmtUnfSF = log(BsmtUnfSF_Z^2),
         FirstFloorSF = log(FirstFloorSF_Z^2),
         GrLivArea = log(GrLivArea_Z^2),
         Rooms = log(Rooms_Z^2),
         GarageCars = log(GarageCars_Z^2),
         GarageArea = log(GarageArea_Z^2)) %>%
  select(SalePrice, OverallQual, LotArea, BsmtUnfSF, FirstFloorSF, GrLivArea, Rooms,
         GarageCars, GarageArea)


str(train.regression)
summary(train.regression)

hist.1 <- ggplot(train.regression, aes(SalePrice)) +
  geom_histogram()
hist.2 <- ggplot(train.regression, aes(LotArea)) +
  geom_histogram()
hist.3 <- ggplot(train.regression, aes(BsmtUnfSF)) +
  geom_histogram()
hist.4 <- ggplot(train.regression, aes(FirstFloorSF)) +
  geom_histogram()
hist.5 <- ggplot(train.regression, aes(GrLivArea)) +
  geom_histogram()
hist.6 <- ggplot(train.regression, aes(Rooms)) +
  geom_histogram()
hist.7 <- ggplot(train.regression, aes(GarageCars)) +
  geom_histogram()
hist.8 <- ggplot(train.regression, aes(GarageArea)) +
  geom_histogram()
hist.9 <- ggplot(train.regression, aes(OverallQual)) +
  geom_histogram()

grid.arrange(hist.1, hist.2, hist.3, hist.4, hist.5, hist.6, hist.7, hist.8, hist.9)

# Test, Train Split
set.seed(222333)
splitIndex <- createDataPartition(train.regression$SalePrice,
                                  p = .6,
                                  list = F,
                                  times = 1)

gbm.train.1 <- train.regression

gbm.label <- train.regression$SalePrice
gbm.train.1$SalePrice <- NULL

# 10-fold Cross Validation
cv.10.folds <- createMultiFolds(gbm.label, k = 10, times = 10)

metric <- "RMSE"

objControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10,
                           index = cv.10.folds)

# Parameter Grid Search
gbm.grid.1 <- expand.grid(interaction.depth = c(1, 2, 4, 8),
                          shrinkage = c(0.1, 0.01, 0.001),
                          n.trees = c(100, 200, 400, 800),
                          n.minobsinnode = c(1, 2, 4, 8))

# Best Tune
gbm.grid.1.best <- expand.grid(interaction.depth = 8,
                               shrinkage = 0.01,
                               n.trees = 800,
                               n.minobsinnode = 1)

# Model Training
c1 <- makeCluster(4, type = "SOCK")
registerDoSNOW(c1)

set.seed(123456)
gbm.1.cv.1 <- train(x = gbm.train.1, y = gbm.label,
                    distribution = "gaussian",
                    method = "gbm",
                    trControl = objControl,
                    tuneGrid = gbm.grid.1.best,
                    metric = metric,
                    preProcess = c("center", "scale"))

stopCluster(c1)

summary(gbm.1.cv.1)
plot(varImp(gbm.1.cv.1))
gbm.1.cv.1

test.regression <- test %>%
  rename(FirstFloorSF = X1stFlrSF, SecondFloorSF = X2ndFlrSF,
         Bedroom = BedroomAbvGr, Kitchen = KitchenAbvGr,
         ThreeSeasonPorch = X3SsnPorch, Rooms = TotRmsAbvGrd) %>%
  select(LotArea, OverallQual, BsmtUnfSF, FirstFloorSF,
         GrLivArea, Rooms, GarageCars, GarageArea, YearBuilt) %>%
  mutate(OverallQual_Z = (OverallQual - mean(OverallQual))/sd(OverallQual),
         LotArea_Z = (LotArea - mean(LotArea))/sd(LotArea),
         BsmtUnfSF_Z = (BsmtUnfSF - mean(BsmtUnfSF))/sd(BsmtUnfSF),
         FirstFloorSF_Z = (FirstFloorSF - mean(FirstFloorSF))/sd(FirstFloorSF),
         GrLivArea_Z = (GrLivArea - mean(GrLivArea))/sd(GrLivArea),
         Rooms_Z = (Rooms - mean(Rooms))/sd(Rooms),
         GarageCars_Z = (GarageCars - mean(GarageCars))/sd(GarageCars),
         GarageArea_Z = (GarageArea - mean(GarageArea))/sd(GarageArea)) %>%
  mutate(OverallQual = log(OverallQual_Z^2),
         LotArea = log(LotArea_Z^2),
         BsmtUnfSF = log(BsmtUnfSF_Z^2),
         FirstFloorSF = log(FirstFloorSF_Z^2),
         GrLivArea = log(GrLivArea_Z^2),
         Rooms = log(Rooms_Z^2),
         GarageCars = log(GarageCars_Z^2),
         GarageArea = log(GarageArea_Z^2)) %>%
  select(OverallQual, LotArea, BsmtUnfSF, FirstFloorSF, GrLivArea, Rooms,
         GarageCars, GarageArea)

preds.GBM <- predict(gbm.1.cv.1, test.regression)
test.submission <- data.frame(Id = test$Id, SalePrice = preds.GBM)

str(test.submission)

write.csv(test.submission, file = "Test.Submission.GBM.Continuous.csv")

#-------------------------------------------------------------------------------------------
#                           Random Forest
#-------------------------------------------------------------------------------------------

train$MSSubClass <- as.factor(train$MSSubClass)
train$LotShape <- as.factor(train$LotShape)
train$LandContour <- as.factor(train$LandContour)
train$LotConfig <- as.factor(train$LotConfig)
train$Neighborhood <- as.factor(train$Neighborhood)
train$Condition1 <- as.factor(train$Condition1)
train$BldgType <- as.factor(train$BldgType)
train$RoofStyle <- as.factor(train$RoofStyle)
train$Exterior1st <- as.factor(train$Exterior1st)
train$Exterior2nd <- as.factor(train$Exterior2nd)
train$OverallCond <- as.factor(train$OverallCond)
train$OverallQual <- as.factor(train$OverallQual)
train$BedroomAbvGr <- as.factor(train$BedroomAbvGr)
train$GarageCars <- as.factor(train$GarageCars)

train.rf <- train %>%
  rename(Rooms = BedroomAbvGr) %>%
  select(SalePrice, MSSubClass, LotArea,
         LotShape, LandContour, LotConfig, Neighborhood,
         Condition1, BldgType, OverallQual,
         OverallCond, RoofStyle,
         Exterior1st, Exterior2nd, GrLivArea,
         Rooms, GarageCars, GarageArea, YearBuilt) %>%
  mutate(SalePrice_z = (SalePrice - mean(SalePrice))/sd(SalePrice),
         LotArea_z = (LotArea - mean(LotArea))/sd(LotArea),
         GrLivArea_z = (GrLivArea - mean(GrLivArea))/sd(GrLivArea),
         GarageArea_z = (GarageArea - mean(GarageArea))/sd(GarageArea)) %>%
  filter(SalePrice_z >= -3.0 & LotArea_z <= 3.0,
         LotArea_z >= -3.0 & SalePrice_z <= 3.0,
         GrLivArea_z >= -3.0 & GrLivArea_z <= 3.0,
         GarageArea_z >= -3.0 & GarageArea_z <= 3.0) %>%
  select(-c(SalePrice_z, LotArea_z, GrLivArea_z, GarageArea_z)) %>%
  mutate(DecadeBuilt = cut(YearBuilt, breaks = 14, labels = c(1:14))) %>%
  select(SalePrice, MSSubClass, LotArea,
         LotShape, LandContour, LotConfig, Neighborhood,
         Condition1, BldgType, OverallQual,
         OverallCond, RoofStyle,
         Exterior1st, Exterior2nd, GrLivArea,
         Rooms, GarageCars, GarageArea, DecadeBuilt)

str(train.rf)

num.train.rf <- train %>%
  select(SalePrice, LotArea, GrLivArea, GarageArea) %>%
  mutate(SalePrice_z = (SalePrice - mean(SalePrice))/sd(SalePrice),
         LotArea_z = (LotArea - mean(LotArea))/sd(LotArea),
         GrLivArea_z = (GrLivArea - mean(GrLivArea))/sd(GrLivArea),
         GarageArea_z = (GarageArea - mean(GarageArea))/sd(GarageArea)) %>%
  filter(SalePrice_z >= -3.0 & LotArea_z <= 3.0,
         LotArea_z >= -3.0 & SalePrice_z <= 3.0,
         GrLivArea_z >= -3.0 & GrLivArea_z <= 3.0,
         GarageArea_z >= -3.0 & GarageArea_z <= 3.0) %>%
  select(-c(SalePrice_z, LotArea_z, GrLivArea_z, GarageArea_z))


#-------------------------------------------------------------------------------#
# Numeric Variables
ggplot(train.rf, aes(SalePrice)) +
  geom_histogram()
ggplot(train.rf, aes(LotArea)) +
  geom_histogram()
ggplot(train.rf, aes(GrLivArea)) +
  geom_histogram()
ggplot(train.rf, aes(GarageArea)) +
  geom_histogram()

pairs(train.rf[, c("SalePrice", "LotArea", "GrLivArea", "GarageArea")])

# Factor Variables
ggplot(train.rf, aes(MSSubClass)) +
  geom_bar()
ggplot(train.rf, aes(group = MSSubClass, MSSubClass, SalePrice)) +
  boxplot()
ggplot(train.rf, aes(group = DecadeBuilt, DecadeBuilt, SalePrice)) +
  geom_boxplot()
#-------------------------------------------------------------------------------#

rf.train.1 <- num.train.rf

# 10 Fold CV
cv.10.folds <- createMultiFolds(rf.train.1$SalePrice,
                                k = 10, 
                                times = 10)

objControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10,
                           index = cv.10.folds)


# Parameters
grid.search <- expand.grid(interaction.depth = 1,
                           shrinkage = 0.001,
                           n.trees = 1001,
                           n.minobsinnode = 1)

# Model train
time1 <- Sys.time()
c1 <- makeCluster(4, type = "SOCK")
registerDoSNOW(c1)

rf.1.cv.1 <- train(SalePrice ~., data = rf.train.1,
                   method = "gbm",
                   metric = "RMSE",
                   trControl = objControl,
                   tuneGrid = grid.search)

stopCluster(c1)
time2 <- Sys.time()
elapsedTime <- time2 - time1
elapsedTime

#Evaluate
rf.1.cv.1
plot(varImp(rf.1.cv.1), top = 10)
summary(rf.1.cv.1)

# Test
num.test.rf <- test %>%
  select(LotArea, GrLivArea, GarageArea)


preds.rf <- predict(rf.1.cv.1, num.test.rf)
test.submission <- data.frame(Id = test$Id, SalePrice = preds.rf)

write.csv(test.submission, file = "test.submission.RF.csv")

