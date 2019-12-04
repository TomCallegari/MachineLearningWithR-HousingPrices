#
#
#
#

#------------------- ENVIRONMENT SETUP -------------------#

# Import libraries
library(mlbench)
library(MASS)
library(psych)
library(gridExtra)
library(doSNOW)
library(caret)
library(caretEnsemble)
library(Boruta)
library(kernlab)
library(randomForest)
library(gbm)
library(neuralnet)
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

#------------------- COMBINE DATASETS -------------------#

# Create Train & Test labels
train_raw <- train_raw %>%
  mutate(dataPartition = "Train")

test_raw <- test_raw %>%
  mutate(dataPartition = "Test",
         SalePrice = "NA") # Note: A SalePrice column must be addeed to test for binding to train

# Bind Train & Test
test_raw$SalePrice <- as.integer(test_raw$SalePrice)

data_full <- bind_rows(train_raw, test_raw)

#------------------- DATA CLEANING & IMPUTATION -------------------#

# MSSubClass
data_full$MSSubClass <- as.factor(data_full$MSSubClass)
data_full$MSSubClass <- plyr::revalue(data_full$MSSubClass, c("150" = "160"))

# Impute MSZoning
data_full$MSZoning[is.na(data_full$MSZoning)] <- "RL"
data_full$MSZoning <- as.factor(data_full$MSZoning)

# LotFrontage
data_full$LotFrontage[is.na(data_full$LotFrontage)] <- median(data_full$LotFrontage, na.rm = T)
data_full$LotFrontage <- as.numeric(data_full$LotFrontage)

# LotArea
data_full$LotArea <- as.numeric(data_full$LotArea)

# Street
data_full$Street <- as.factor(data_full$Street)

# Alley
data_full$Alley[is.na(data_full$Alley)] <- "None"
data_full$Alley <- as.factor(data_full$Alley)

# LotShape
data_full$LotShape <- as.factor(data_full$LotShape)

# LandContour
data_full$LandContour <- as.factor(data_full$LandContour)

# Utilities

# LotConfig
data_full$LotConfig <- as.factor(data_full$LotConfig)

# LandSlope
data_full$LandSlope <- as.factor(data_full$LandSlope)

# Neighborhood becomes neighborhoodValues
median_neighborhood_value <- data_full %>%
  filter(dataPartition == "Train") %>%
  select(Neighborhood, SalePrice) %>%
  group_by(Neighborhood) %>%
  summarise(medianSalePrice = median(SalePrice)) %>%
  mutate(neighborhoodValues = cut(medianSalePrice, breaks = 10, labels = c(1:10))) %>%
  select(Neighborhood, neighborhoodValues)
data_full <- left_join(data_full, median_neighborhood_value, by.x = "Neighborhood", by.y = "Neighborhood")
data_full$neighborhoodValues<- as.numeric(data_full$neighborhoodValues)

# Condition1 - Combine street, railroad and off-site feature levels 
data_full$Condition1 <- as.factor(data_full$Condition1)
data_full$Condition1 <- plyr::revalue(data_full$Condition1, c("Artery" = "Street",
                                                              "Feedr" = "Street",
                                                              "RRNn" = "Railroad",
                                                              "RRAn" = "Railroad",
                                                              "PosN" = "Park",
                                                              "PosA" = "Park",
                                                              "RRNe" = "Railroad",
                                                              "RRAe" = "Railroad"))
# Condition2
data_full$Condition2 <- as.factor(data_full$Condition2)
data_full$Condition2 <- plyr::revalue(data_full$Condition2, c("Artery" = "Street",
                                                              "Feedr" = "Street",
                                                              "RRNn" = "Railroad",
                                                              "RRAn" = "Railroad",
                                                              "PosN" = "Park",
                                                              "PosA" = "Park",
                                                              "RRAe" = "Railroad"))

# BldgType
data_full$BldgType <- as.factor(data_full$BldgType)
data_full$BldgType <- plyr::revalue(data_full$BldgType, c("2fmCon" = "Duplex",
                                                          "Twnhs" = "Townhouse",
                                                          "TwnhsE" = "Townhouse"))

# HouseStyle
data_full$HouseStyle <- as.factor(data_full$HouseStyle)
data_full$HouseStyle <- plyr::revalue(data_full$HouseStyle, c("1.5Fin" = "1Story",
                                                              "1.5Unf" = "1Story",
                                                              "2.5Fin" = "2Story",
                                                              "2.5Unf" = "2Story",
                                                              "SFoyer" = "Split",
                                                              "SLvl" = "Split"))
# OverallQual & OverallCond
data_full$OverallQual <- as.numeric(data_full$OverallQual)
data_full$OverallCond <- as.numeric(data_full$OverallCond)

# data_full <- data_full %>%
#   mutate(OverallQual = ordered(OverallQual, levels = c(1:10)),
#          OverallCond = ordered(OverallCond, levels = c(1:10)))

data_full$OveralQual <- as.numeric(data_full$OverallQual)
data_full$OveralCond <- as.numeric(data_full$OverallCond)

# YearBuilt becomes DecadeBuilt
data_full <- data_full %>%
  mutate(DecadeBuilt = cut(YearBuilt,
                                breaks = c(-Inf, 1900,
                                           1910, 1920,
                                           1930, 1940,
                                           1950, 1960,
                                           1970, 1980,
                                           1990, 2000,
                                           2010),
                                labels = c(1:12)))

data_full$YearBuilt <- as.numeric(data_full$YearBuilt)

# YearRemodAdd becomes DecadeRemod
data_full <- data_full %>%
  mutate(DecadeRemod = cut(YearRemodAdd,
                           breaks = c(-Inf, 1950,
                                      1960, 1970,
                                      1980, 1990,
                                      2000, 2010),
                           labels = c("None", 1:6)))

data_full$YearRemodAdd <- as.factor(data_full$YearRemodAdd)

# RoofStyle
data_full$RoofStyle <- as.factor(data_full$RoofStyle)

# RoofMatl
data_full$RoofMatl <- as.factor(data_full$RoofMatl)

# Exterior1st
data_full$Exterior1st[is.na(data_full$Exterior1st)] <- "VinylSd"
data_full$Exterior1st <- as.factor(data_full$Exterior1st)
data_full$Exterior1st <- plyr::revalue(data_full$Exterior1st, c("Wd Sdng" = "WdSdng"))

# Exterior2nd
data_full$Exterior2nd[is.na(data_full$Exterior2nd)] <- "VinylSd"
data_full$Exterior2nd <- as.factor(data_full$Exterior2nd)
data_full$Exterior2nd <- plyr::revalue(data_full$Exterior2nd, c("Wd Sdng" = "Wdsdng",
                                                                "Brk Cmn" = "BrkComm"))
# MasVnrType
data_full$MasVnrType[is.na(data_full$MasVnrType)] <- "None"
data_full$MasVnrType <- as.factor(data_full$MasVnrType)

# MasVnrArea
data_full$MasVnrArea[is.na(data_full$MasVnrArea)] <- 0

# ExterQual
data_full$ExterQual <- as.factor(data_full$ExterQual)
data_full$ExterQual[is.na(data_full$ExterQual)] <- "TA"
data_full$ExterQual <- plyr::revalue(data_full$ExterQual, c("Ex" = "4",
                                                            "Gd" = "3",
                                                            "TA" = "2",
                                                            "Fa" = "1"))
data_full$ExterQual <- as.numeric(data_full$ExterQual)

# data_full <- data_full %>%
#   mutate(ExterQual = ordered(ExterQual, levels = c(1:4)))

# ExterCond
data_full$ExterCond <- as.factor(data_full$ExterCond)
data_full$ExterCond <- plyr::revalue(data_full$ExterCond, c("Ex" = "5",
                                                            "Gd" = "4",
                                                            "TA" = "3",
                                                            "Fa" = "2",
                                                            "Po" = "1"))
data_full$ExterCond <- as.numeric(data_full$ExterCond)

# data_full <- data_full %>%
#   mutate(ExterCond = ordered(ExterCond, levels = c(1:5)))

# Foundation
data_full$Foundation <- as.factor(data_full$Foundation)

# BsmtQual
data_full$BsmtQual[is.na(data_full$BsmtQual)] <- "0" # "None"
data_full$BsmtQual <- plyr::revalue(data_full$BsmtQual, c("Fa" = "2",
                                                          "TA" = "3",
                                                          "Gd" = "4",  
                                                          "Ex" = "5"))
data_full$BsmtQual <- as.numeric(data_full$BsmtQual)

# data_full <- data_full %>%
#   mutate(BsmtQual = ordered(BsmtQual, levels = c("None", 2:5)))

# BsmtCond
data_full$BsmtCond[is.na(data_full$BsmtCond)] <- "0" # "None"
data_full$BsmtCond <- plyr::revalue(data_full$BsmtCond, c("Gd" = "5",
                                                          "TA" = "4",
                                                          "Fa" = "3",
                                                          "Po" = "2"))
data_full$BsmtCond <- as.numeric(data_full$BsmtCond)

# data_full <- data_full %>%
#   mutate(BsmtCond = ordered(BsmtCond, levels = c("None", 2:5)))

# BsmtExposure
data_full$BsmtExposure[is.na(data_full$BsmtExposure)] <- "0" # "None"
data_full$BsmtExposure <- plyr::revalue(data_full$BsmtExposure, c("Gd" = "5",
                                                                  "Av" = "4",
                                                                  "Mn" = "3",
                                                                  "No" = "2"))
data_full$BsmtExposure <- as.numeric(data_full$BsmtExposure)

# data_full <- data_full %>%
#   mutate(BsmtExposure = ordered(BsmtExposure, levels = c("None", 2:5)))

# BsmtFinType1
data_full$BsmtFinType1[is.na(data_full$BsmtFinType1)] <- "None"
data_full$BsmtFinType1 <- as.factor(data_full$BsmtFinType1)

# Impute BsmtFinSF1
data_full$BsmtFinSF1[is.na(data_full$BsmtFinSF1)] <- 0

# BsmtFinType2
data_full$BsmtFinType2[is.na(data_full$BsmtFinType2)] <- "None"
data_full$BsmtFinType2 <- as.factor(data_full$BsmtFinType2)

# Impute BsmtFinSF2
data_full$BsmtFinSF2[is.na(data_full$BsmtFinSF2)] <- 0

# Impute BsmtUnfSF
data_full$BsmtUnfSF[is.na(data_full$BsmtUnfSF)] <- 0

# Impute TotalBsmtSF
data_full$TotalBsmtSF[is.na(data_full$TotalBsmtSF)] <- 0

# Heating (Near zero variance)
data_full$Heating <- as.factor(data_full$Heating)

# HeatingQC
data_full$HeatingQC <- plyr::revalue(data_full$HeatingQC, c("Ex" = "5",
                                                            "Gd" = "4",
                                                            "TA" = "3",
                                                            "Fa" = "2",
                                                            "Po" = "1"))
data_full$HeatingQC <- as.numeric(data_full$HeatingQC)

# data_full <- data_full %>%
#   mutate(HeatingQC = ordered(HeatingQC, levels = c(1:5)))

# CentralAir
data_full$CentralAir <- as.factor(data_full$CentralAir)

# Electrical
data_full$Electrical[is.na(data_full$Electrical)] <- "SBrkr"
data_full$Electrical <- ifelse(data_full$Electrical == "FuseA", "Other",
                               ifelse(data_full$Electrical == "FuseF", "Other",
                                      ifelse(data_full$Electrical == "FuseP", "Other",
                                             ifelse(data_full$Electrical == "Mix", "Other", "SBrkr"))))
data_full$Electrical <- as.factor(data_full$Electrical)

# FirstFloorSF & SecondFloorSF
data_full <- data_full %>%
  rename(FirstFloorSF = X1stFlrSF,
         SecondFloorSF = X2ndFlrSF)

data_full$FirstFloorSF <- as.numeric(data_full$FirstFloorSF)
data_full$SecondFloorSF <- as.numeric(data_full$SecondFloorSF)

# LowQualFinSF
data_full$LowQualFinSF <- as.numeric(data_full$LowQualFinSF)

# GrLivArea
data_full$GrLivArea <- as.numeric(data_full$GrLivArea)

# BsmtFullBath
data_full$BsmtFullBath[is.na(data_full$BsmtFull)] <- 0

# BsmtHalfBath
data_full$BsmtHalfBath[is.na(data_full$BsmtHalfBath)] <- 0

# FullBath
data_full$FullBath <- as.numeric(data_full$FullBath)

# HalfBath
data_full$HalfBath <- as.numeric(data_full$HalfBath)

# Bedroom
data_full <- data_full %>%
  rename(Bedroom = BedroomAbvGr)

data_full$Bedroom <- as.numeric(data_full$Bedroom)

# Kitchen
data_full <- data_full %>%
  rename(Kitchen = KitchenAbvGr)

data_full$Kitchen <- as.numeric(data_full$Kitchen)

# KitchenQual
data_full$KitchenQual <- as.factor(data_full$KitchenQual)
data_full$KitchenQual[is.na(data_full$KitchenQual)] <- "TA"
data_full$KitchenQual <- plyr::revalue(data_full$KitchenQual, c("Fa" = "1",
                                                                "TA" = "2",
                                                                "Gd" = "3",     
                                                                "Ex" = "4"))
data_full$KitchenQual <- as.numeric(data_full$KitchenQual)

# data_full <- data_full %>%
#   mutate(KitchenQual = ordered(KitchenQual, levels = c(1:4)))

# TotRmsAbvGrd
data_full$TotRmsAbvGrd <- as.factor(data_full$TotRmsAbvGrd)
data_full <- data_full %>%
  rename(TotalRooms = TotRmsAbvGrd)

# Functional
data_full$Functional[is.na(data_full$Functional)] <- "Typ"
data_full$Functional <- as.factor(data_full$Functional)

# Fireplaces
data_full$Fireplaces <- as.numeric(data_full$Fireplaces)

# FireplaceQu
data_full$FireplaceQu[is.na(data_full$FireplaceQu)] <- "0" # "None"
data_full$FireplaceQu <- plyr::revalue(data_full$FireplaceQu, c("Ex" = "6",
                                                                "Gd" = "5",
                                                                "TA" = "4",
                                                                "Fa" = "3",
                                                                "Po" = "2"))
data_full$FireplaceQu <- as.numeric(data_full$FireplaceQu)

# data_full <- data_full %>%
#   mutate(FireplaceQu = ordered(FireplaceQu, levels = c("None", 2:6)))

# GarageType
data_full$GarageType[is.na(data_full$GarageType)] <- "None"
data_full$GarageType <- as.factor(data_full$GarageType)

# GarageYrBlt becomes GarageDecadeBuilt
data_full <- data_full %>%
  mutate(GarageYrBlt = replace(GarageYrBlt, GarageYrBlt == 2207, 2007))
data_full$GarageYrBlt[is.na(data_full$GarageYrBlt)] <- 0
data_full <- data_full %>%
  mutate(GarageDecadeBuilt = cut(GarageYrBlt, 
                                 breaks = c(-Inf, 0, 
                                            1895, 1905,
                                            1915, 1925,
                                            1935, 1945,
                                            1955, 1965,
                                            1975, 1985,
                                            1995, 2005,
                                            2010),
                                 labels = c("None", 1:13)))

# GarageFinish
data_full$GarageFinish[is.na(data_full$GarageFinish)] <- "None"
data_full$GarageFinish <- as.factor(data_full$GarageFinish)

# GarageCars
data_full$GarageCars[is.na(data_full$GarageCars)] <- 0

# GarageArea
data_full$GarageArea[is.na(data_full$GarageArea)] <- 0

# GarageQual
data_full$GarageQual[is.na(data_full$GarageQual)] <- "0" # "None"
data_full$GarageQual <- plyr::revalue(data_full$GarageQual, c("Ex" = "6",
                                                              "Gd" = "5",
                                                              "TA" = "4",
                                                              "Fa" = "3",
                                                              "Po" = "2"))
data_full$GarageQual <- as.numeric(data_full$GarageQual)

# data_full <- data_full %>%
#   mutate(GarageQual = ordered(GarageQual, levels = c("None", 2:6)))

# GarageCond
data_full$GarageCond[is.na(data_full$GarageCond)] <- "0" # "None"
data_full$GarageCond <- plyr::revalue(data_full$GarageCond, c("Ex" = "6",
                                                              "Gd" = "5",
                                                              "TA" = "4",
                                                              "Fa" = "3",
                                                              "Po" = "2"))
data_full$GarageCond <- as.numeric(data_full$GarageCond)

# data_full <- data_full %>%
#   mutate(GarageCond = ordered(GarageCond, levels = c("None", 2:6)))

# PavedDrive
data_full$PavedDrive <- as.factor(data_full$PavedDrive)

# WoodDeckSF
data_full$WoodDeckSF <- as.numeric(data_full$WoodDeckSF)

# OpenPorchSF
data_full$OpenPorchSF <- as.numeric(data_full$OpenPorchSF)

# EncolsedPorch
data_full$EnclosedPorch <- as.numeric(data_full$EnclosedPorch)

# ThreeSeasonPorchSF
data_full <- data_full %>%
  rename(ThreeSeasonPorch = X3SsnPorch)
data_full$ThreeSeasonPorch <- as.numeric(data_full$ThreeSeasonPorch)

# ScreenPorch
data_full$ScreenPorch <- as.numeric(data_full$ScreenPorch)

# PoolArea 
data_full$PoolArea <- as.numeric(data_full$PoolArea)

# PoolQC
data_full$PoolQC <- as.factor(data_full$PoolQC)

# Fence
data_full$Fence[is.na(data_full$Fence)] <- "None"
data_full$Fence <- as.factor(data_full$Fence)

# MiscFeature
data_full$MiscFeature[is.na(data_full$MiscFeature)] <- "None"
data_full$MiscFeature <- as.factor(data_full$MiscFeature)

# MiscVal
data_full$MiscVal <- as.numeric(data_full$MiscVal)

# MoSold
data_full$MoSold <- as.factor(data_full$MoSold)

# YrSold 
data_full$YrSold <- as.numeric(data_full$YrSold)

# SaleType
data_full$SaleType[is.na(data_full$SaleType)] <- "WD"
data_full$SaleType <- as.factor(data_full$SaleType)

# SaleCondition
data_full$SaleCondition <- plyr::revalue(data_full$SaleCondition, c("AdjLand" = "Combine",
                                                                    "Alloca" = "Combine",
                                                                    "Family" = "Combine"))
data_full$SaleCondition <- as.factor(data_full$SaleCondition)

#------------------- FEATURE ENGINEERING -------------------#

# houseAge
data_full <- data_full %>%
  mutate(houseAge = (YrSold + 1) - YearBuilt)

data_full$newHouse <- ifelse(data_full$YrSold == data_full$YearBuilt, 1, 0)

# StructureSF, OpenPorchSF, FeatureCount, TotalRooms, TotalBathrooms, avgRoomSize, PorchSF
data_full <- data_full %>%
  mutate(StructureSF = TotalBsmtSF + GrLivArea + GarageArea + WoodDeckSF + 
           OpenPorchSF + EnclosedPorch + ThreeSeasonPorch + ScreenPorch,
         TotalBathrooms = BsmtFullBath + BsmtHalfBath + FullBath + HalfBath,
         TotalRooms = TotalBathrooms + Bedroom + Kitchen,
         FeatureCount = TotalRooms + Fireplaces + GarageCars + TotalBathrooms,
         avgRoomSize = StructureSF / TotalRooms,
         PorchSF = WoodDeckSF + OpenPorchSF + EnclosedPorch + ThreeSeasonPorch + ScreenPorch,
         sizeRatio = StructureSF / LotArea,
         avgFeatureSize = StructureSF / FeatureCount,
         avgBathSize = StructureSF / TotalBathrooms,
         yardSize = LotArea - (FirstFloorSF + GarageArea + PoolArea))

# median_Bathrooms
median_Bathrooms <- data_full %>%
  select(Neighborhood, TotalBathrooms) %>%
  group_by(Neighborhood) %>%
  summarise(medianBathrooms = median(TotalBathrooms)) %>%
  mutate(medianNeighborhoodBathrooms = cut(medianBathrooms, breaks = 10, labels = c(1:10))) %>%
  select(Neighborhood, medianNeighborhoodBathrooms)

data_full <- left_join(data_full, median_Bathrooms, by.x = "Neighborhood", by.y = "Neighborhood") 
data_full$medianNeighborhoodBathrooms <- as.numeric(data_full$medianNeighborhoodBathrooms)

# medianNeighborhoodRooms
median_Rooms <- data_full %>%
  select(Neighborhood, TotalRooms) %>%
  group_by(Neighborhood) %>%
  summarise(medianRooms = median(TotalRooms)) %>%
  mutate(medianNeighborhoodRooms = cut(medianRooms, breaks = 10, labels = c(1:10))) %>%
  select(Neighborhood, medianNeighborhoodRooms)

data_full <- left_join(data_full, median_Rooms, by.x = "Neighborhood", by.y = "Neighborhood")
data_full$medianNeighborhoodRooms <- as.numeric(data_full$medianNeighborhoodRooms)

# medianNeighborhoodFeatureCount
median_FeatureCount <- data_full %>%
  select(Neighborhood, FeatureCount) %>%
  group_by(Neighborhood) %>%
  summarise(medianFeatureCount = median(FeatureCount)) %>%
  mutate(medianNeighborhoodFeatureCount = cut(medianFeatureCount, breaks = 10, labels = c(1:10))) %>%
  select(Neighborhood, medianNeighborhoodFeatureCount)

data_full <- left_join(data_full, median_FeatureCount, by.x = "Neighborhood", by.y = "Neighborhood")
data_full$medianNeighborhoodFeatureCount <- as.numeric(data_full$medianNeighborhoodFeatureCount)

# medianNeighborhoodStructureSF
median_Structure_SF <- data_full %>%
  select(Neighborhood, StructureSF) %>%
  group_by(Neighborhood) %>%
  summarise(medianStructureSF = median(StructureSF)) %>%
  mutate(medianNeighborhoodStructureSF = cut(medianStructureSF, breaks = 10, labels = c(1:10))) %>%
  select(Neighborhood, medianNeighborhoodStructureSF)

data_full <- left_join(data_full, median_Structure_SF, by.x = "Neighborhood", by.y = "Neighborhood")
data_full$medianNeighborhoodStructureSF <- as.numeric(data_full$medianNeighborhoodStructureSF)

# medianNeighborhoodLotArea
median_LotArea <- data_full %>%
  select(Neighborhood, LotArea) %>%
  group_by(Neighborhood) %>%
  summarise(medianLotArea = median(LotArea)) %>%
  mutate(medianNeighborhoodLotArea = cut(medianLotArea, breaks = 10, labels = c(1:10))) %>%
  select(Neighborhood, medianNeighborhoodLotArea)

data_full <- left_join(data_full, median_LotArea, by.x = "Neighborhood", by.y = "Neighborhood")
data_full$medianNeighborhoodLotArea <- as.numeric(data_full$medianNeighborhoodLotArea)

# hasPorch
data_full$hasPorch <- ifelse(data_full$PorchSF > 0, 1, 0)
data_full$hasPorch <- as.factor(data_full$hasPorch)

# hasGarage
data_full$hasGarage <- ifelse(data_full$GarageQual == 0, 0, 1)
data_full$hasGarage <- as.factor(data_full$hasGarage)

# hasPool
data_full$hasPool <- ifelse(data_full$PoolArea == 0, 0, 1)
data_full$hasPool <- as.factor(data_full$hasPool)

# hasFireplace
data_full$hasFireplace <- ifelse(data_full$Fireplaces > 0, 1, 0)
data_full$hasFireplace <- as.factor(data_full$hasFireplace)

# hasBasement
data_full$hasBasement <- ifelse(data_full$TotalBsmtSF == 0, 0, 1)
data_full$hasBasement <- as.factor(data_full$hasBasement)

# Renovated
data_full$Renovated <- ifelse(data_full$YearRemodAdd == data_full$YearBuilt, 0, 1)
data_full$Renovated <- as.factor(data_full$Renovated)

# Convert variables to factor
data_full$Utilities <- as.factor(data_full$Utilities)
data_full$Neighborhood <- as.factor(data_full$Neighborhood)
data_full$YrSold <- as.factor(data_full$YrSold)
data_full$MoSold <- as.factor(data_full$MoSold)

# Remove PoolQC & Utilities
data_full$PoolQC <- NULL
data_full$Utilities <- NULL

#------------------- WRITE AND LOAD CLEANED DATASET -------------------#

# Write csv
write.csv(data_full, file = "data_full.csv")

# Load csv
data_full <- read.csv("data_full.csv",
                      header = T,
                      stringsAsFactors = F)
data_full$X <- NULL

#------------------- DUMMY VARIABLES AND NEAR ZERO VARIANCE FEATURES -------------------#

# Pull out Id and dataPartition
dataIndexVars <- data_full %>%
  select(Id, dataPartition, SalePrice)

data_full <- data_full %>%
  select(-c(Id, dataPartition, SalePrice))

# DummyVars for categorical
dmy.train <- dummyVars(" ~ .", data = data_full)
data_full <- data.frame(predict(dmy.train, newdata = data_full))

# Remove Near Zero Variance columns
nzv_cols <- nearZeroVar(data_full, freqCut = 95/5)
if (length(nzv_cols) > 0) data_full <- data_full[, -nzv_cols]

# Bind dataIndexVars back onto data_full
model_data <- bind_cols(dataIndexVars, data_full)

#------------------- RECURSIVE FEATURE ELIMINATION -------------------#

rfeTrain <- model_data %>%
  filter(dataPartition == "Train") %>%
  select(-c(Id, dataPartition))

rfe_Control <- rfeControl(functions = rfFuncs, 
                          method = "cv", 
                          number = 5)

set.seed(1)
rfeResults <- rfe(rfeTrain[, 2:147],
                  rfeTrain$SalePrice,
                  rfeControl = rfe_Control,
                  metric = "RMSE",
                  maximize = F)

rfeVariables <- predictors(rfeResults)

model_data <- model_data[, rfeVariables]

model_data <- bind_cols(dataIndexVars, model_data)

#------------------- SPLIT INTO TRAIN & TEST -------------------#

train <- model_data %>%
  filter(dataPartition == "Train") %>%
  select(-c(Id, dataPartition))

plot_correlation(train)

test <- model_data %>%
  filter(dataPartition == "Test") %>%
  select(-c(dataPartition))

test.pred <- test %>%
  select(-c(Id))

#------------------- MODEL(S) SETUP -------------------#

# Create folds index for CV
set.seed(3)
cv_1_folds_1 <- createMultiFolds(train$SalePrice,
                                 k = 5,
                                 times = 5)
set.seed(4)
objControl <- trainControl(method = "repeatedcv",
                           number = 5, 
                           repeats = 5,
                           search = "grid",
                           savePredictions = T,
                           index = cv_1_folds_1,
                           verboseIter = T)

# Parameter Grids
GBM.grid <- expand.grid(interaction.depth = 8,
                        shrinkage = 0.01,
                        n.trees = 3000,
                        n.minobsinnode = 2)

RF.grid <- expand.grid(mtry = 80)

#------------------- MODELS ------------------#

time1 <- Sys.time()

set.seed(5)
GBM.model <- train(SalePrice ~ ., 
                   data = train,
                   trControl = objControl,
                   metric = "RMSE",
                   method = "gbm",
                   tuneGrid = GBM.grid,
                   preProcess = c("center", "scale"))

summary(GBM.model)
GBM.model

time2 <- Sys.time()
elapsedTime <- time2 - time1
elapsedTime

time1 <- Sys.time()

set.seed(6)                   
RF.model <- train(SalePrice ~ ., 
                  data = train,
                  metric = "RMSE",
                  tuneGrid = RF.grid,
                  trControl = objControl,
                  ntree = 1000,
                  preProcess = c("center", "scale"))

summary(RF.model)
RF.model

time2 <- Sys.time()
elapsedTime <- time2 - time1
elapsedTime

time1 <- Sys.time()

set.seed(7)                   
SVM.model <- train(SalePrice ~ ., 
                   data = train,
                   trControl = objControl,
                   metric = "RMSE",
                   method = "svmRadial",
                   prePRocess = c("center", "scale"))

summary(SVM.model)
SVM.model

time2 <- Sys.time()
elapsedTime <- time2 - time1
elapsedTime

GBM.model
RF.model
SVM.model

#------------------- Model(s) Evaluation -------------------#

combined_models<- list(gbm = GBM.model, 
                       rf = RF.model,
                       svmRadial = SVM.model)

class(combined_models) <- "caretList"

modelCor(resamples(combined_models))
summary(resamples(combined_models))

#------------------- Weighted Ensemble -------------------#

set.seed(11)
ensembleControl <- trainControl(method = "repeatedcv",
                                number = 5, 
                                repeats = 5,
                                search = "grid",
                                savePredictions = T,
                                index = cv_1_folds_1,
                                verboseIter = T)

# Weighted Ensemble
set.seed(12)
models_ensemble <- caretEnsemble(combined_models,
                                 metric = "RMSE",
                                 trControl = ensembleControl)

# Evaluate
summary(models_ensemble)

#------------------- Prediction & Submission -------------------#

# Numeric centered & scaled (Ensemble - GBM, RF, SVM)
preds <- predict(models_ensemble, newdata = test.pred)

ensemble_data_full <- data.frame(Id = test$Id, SalePrice = preds)
head(ensemble_data_full)

write.csv(scaled_ensemble, file = "ensemble_data_full.csv")








