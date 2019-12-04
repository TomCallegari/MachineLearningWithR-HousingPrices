





#
#
#
#

#------------------- ENVIRONMENT SETUP -------------------#

# Import libraries
library(recipes)
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
library(xgboost)
library(neuralnet)
library(tidyverse)
library(DataExplorer)

# Set working directory
setwd("D://Analytics/Housing Prices")

# Theme for plots
plot_theme <- theme(rect = element_blank(),
                    text = element_text(colour = "#2c3539", size = 14),
                    line = element_line(colour = "#2c3539"),
                    axis.text.x = element_text(colour = "#2c3539", size = 14),
                    axis.text.y = element_text(colour = "#2c3539", size = 14),
                    axis.title = element_text(colour = "#2c3539", size = 14),
                    plot.title = element_text(hjust = 0.5, size = 20),
                    panel.grid.major = element_line(colour = "#687d86", linetype = "dotted"),
                    panel.grid.minor = element_line(colour = "#687d86", linetype = "dotted"),
                    panel.border = element_rect(colour = "#2c3539", size = .5),
                    plot.background = element_blank())

#------------------- LOAD DATA -------------------#

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

#------------------- DATA CLEANING AND IMPUTATION WITH VISUALIZATION -------------------#

# MSSubClass
data_full$MSSubClass <- as.factor(data_full$MSSubClass)
data_full$MSSubClass <- plyr::revalue(data_full$MSSubClass, c("150" = "160"))

ggplot(data_full, aes(MSSubClass)) +
  geom_bar(fill = "#C0C0C0", col = "#2c3539", size = 1.25) +
  ylab("Count") +
  plot_theme 

# Impute MSZoning
data_full$MSZoning[is.na(data_full$MSZoning)] <- "RL"
data_full$MSZoning <- as.factor(data_full$MSZoning)

ggplot(data_full, aes(MSZoning)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme 

# LotFrontage
data_full$LotFrontage[is.na(data_full$LotFrontage)] <- median(data_full$LotFrontage, na.rm = T)
data_full$LotFrontage <- as.numeric(data_full$LotFrontage)

ggplot(data_full, aes(LotFrontage)) +
  geom_histogram(binwidth = 15, fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme 

# LotArea
data_full$LotArea <- as.numeric(data_full$LotArea)

ggplot(data_full, aes(LotArea)) +
  geom_histogram(binwidth = 3500, fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme 

# Street
data_full$Street <- as.factor(data_full$Street)

ggplot(data_full, aes(Street)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme 

# Alley
data_full$Alley[is.na(data_full$Alley)] <- "None"
data_full$Alley <- as.factor(data_full$Alley)

ggplot(data_full, aes(Alley)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme 

# LotShape
data_full$LotShape <- as.factor(data_full$LotShape)

ggplot(data_full, aes(LotShape)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme 

# LandContour
data_full$LandContour <- as.factor(data_full$LandContour)

ggplot(data_full, aes(LandContour)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme 

# Utilities
data_full$Utilities <- as.factor(data_full$Utilities)

ggplot(data_full, aes(Utilities)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme 

# LotConfig
data_full$LotConfig <- as.factor(data_full$LotConfig)

ggplot(data_full, aes(LotConfig)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme 

# LandSlope
data_full$LandSlope <- as.factor(data_full$LandSlope)

ggplot(data_full, aes(LandSlope)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme 

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

ggplot(data_full, aes(Condition1)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme 

# Condition2
data_full$Condition2 <- as.factor(data_full$Condition2)
data_full$Condition2 <- plyr::revalue(data_full$Condition2, c("Artery" = "Street",
                                                              "Feedr" = "Street",
                                                              "RRNn" = "Railroad",
                                                              "RRAn" = "Railroad",
                                                              "PosN" = "Park",
                                                              "PosA" = "Park",
                                                              "RRAe" = "Railroad"))

ggplot(data_full, aes(Condition2)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# BldgType
data_full$BldgType <- as.factor(data_full$BldgType)
data_full$BldgType <- plyr::revalue(data_full$BldgType, c("2fmCon" = "Duplex",
                                                          "Twnhs" = "Townhouse",
                                                          "TwnhsE" = "Townhouse"))

ggplot(data_full, aes(BldgType)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# HouseStyle
data_full$HouseStyle <- as.factor(data_full$HouseStyle)
data_full$HouseStyle <- plyr::revalue(data_full$HouseStyle, c("1.5Fin" = "1Story",
                                                              "1.5Unf" = "1Story",
                                                              "2.5Fin" = "2Story",
                                                              "2.5Unf" = "2Story",
                                                              "SFoyer" = "Split",
                                                              "SLvl" = "Split"))
ggplot(data_full, aes(HouseStyle)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# OverallQual
data_full$OverallQual <- as.numeric(data_full$OverallQual)

ggplot(data_full, aes(OverallQual)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# OverallCond
data_full$OverallCond <- as.numeric(data_full$OverallCond)

ggplot(data_full, aes(OverallCond)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# YearBuilt becomes DecadeBuilt
ggplot(data_full, aes(YearBuilt)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

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

ggplot(data_full, aes(DecadeBuilt)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# YearRemodAdd becomes DecadeRemod
data_full <- data_full %>%
  mutate(DecadeRemod = cut(YearRemodAdd,
                           breaks = c(-Inf, 1950,
                                      1960, 1970,
                                      1980, 1990,
                                      2000, 2010),
                           labels = c("None", 1:6)))

data_full$YearRemodAdd <- as.factor(data_full$YearRemodAdd)

ggplot(data_full, aes(DecadeRemod)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# RoofStyle
data_full$RoofStyle <- as.factor(data_full$RoofStyle)

ggplot(data_full, aes(RoofStyle)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# RoofMatl
data_full$RoofMatl <- as.factor(data_full$RoofMatl)

ggplot(data_full, aes(RoofMatl)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# Exterior1st
data_full$Exterior1st[is.na(data_full$Exterior1st)] <- "VinylSd"
data_full$Exterior1st <- as.factor(data_full$Exterior1st)
data_full$Exterior1st <- plyr::revalue(data_full$Exterior1st, c("Wd Sdng" = "WdSdng"))

ggplot(data_full, aes(Exterior1st)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# Exterior2nd
data_full$Exterior2nd[is.na(data_full$Exterior2nd)] <- "VinylSd"
data_full$Exterior2nd <- as.factor(data_full$Exterior2nd)
data_full$Exterior2nd <- plyr::revalue(data_full$Exterior2nd, c("Wd Sdng" = "Wdsdng",
                                                                "Brk Cmn" = "BrkComm"))

ggplot(data_full, aes(Exterior2nd)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# MasVnrType
data_full$MasVnrType[is.na(data_full$MasVnrType)] <- "None"
data_full$MasVnrType <- as.factor(data_full$MasVnrType)

ggplot(data_full, aes(MasVnrType)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# MasVnrArea
data_full$MasVnrArea[is.na(data_full$MasVnrArea)] <- 0

ggplot(data_full, aes(MasVnrArea)) +
  geom_histogram(binwidth = 25, fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# ExterQual
data_full$ExterQual <- as.factor(data_full$ExterQual)
data_full$ExterQual[is.na(data_full$ExterQual)] <- "TA"
data_full$ExterQual <- plyr::revalue(data_full$ExterQual, c("Ex" = "4",
                                                            "Gd" = "3",
                                                            "TA" = "2",
                                                            "Fa" = "1"))
data_full$ExterQual <- as.numeric(data_full$ExterQual)

ggplot(data_full, aes(ExterQual)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# ExterCond
data_full$ExterCond <- as.factor(data_full$ExterCond)
data_full$ExterCond <- plyr::revalue(data_full$ExterCond, c("Ex" = "5",
                                                            "Gd" = "4",
                                                            "TA" = "3",
                                                            "Fa" = "2",
                                                            "Po" = "1"))
data_full$ExterCond <- as.numeric(data_full$ExterCond)

ggplot(data_full, aes(ExterCond)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# Foundation
data_full$Foundation <- as.factor(data_full$Foundation)

ggplot(data_full, aes(Foundation)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# BsmtQual
data_full$BsmtQual[is.na(data_full$BsmtQual)] <- "0" # "None"
data_full$BsmtQual <- plyr::revalue(data_full$BsmtQual, c("Fa" = "2",
                                                          "TA" = "3",
                                                          "Gd" = "4",  
                                                          "Ex" = "5"))
data_full$BsmtQual <- as.factor(data_full$BsmtQual)

ggplot(data_full, aes(BsmtQual)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# BsmtCond
data_full$BsmtCond[is.na(data_full$BsmtCond)] <- "0" # "None"
data_full$BsmtCond <- plyr::revalue(data_full$BsmtCond, c("Gd" = "5",
                                                          "TA" = "4",
                                                          "Fa" = "3",
                                                          "Po" = "2"))
data_full$BsmtCond <- as.numeric(data_full$BsmtCond)

ggplot(data_full, aes(BsmtCond)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# BsmtExposure
data_full$BsmtExposure[is.na(data_full$BsmtExposure)] <- "0" # "None"
data_full$BsmtExposure <- plyr::revalue(data_full$BsmtExposure, c("Gd" = "5",
                                                                  "Av" = "4",
                                                                  "Mn" = "3",
                                                                  "No" = "2"))
data_full$BsmtExposure <- as.numeric(data_full$BsmtExposure)

ggplot(data_full, aes(BsmtExposure)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# BsmtFinType1
data_full$BsmtFinType1[is.na(data_full$BsmtFinType1)] <- "None"
data_full$BsmtFinType1 <- as.factor(data_full$BsmtFinType1)

ggplot(data_full, aes(BsmtFinType1)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# BsmtFinSF1
data_full$BsmtFinSF1[is.na(data_full$BsmtFinSF1)] <- 0

ggplot(data_full, aes(BsmtFinSF1)) +
  geom_histogram(binwidth = 100, fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# BsmtFinType2
data_full$BsmtFinType2[is.na(data_full$BsmtFinType2)] <- "None"
data_full$BsmtFinType2 <- as.factor(data_full$BsmtFinType2)

ggplot(data_full, aes(BsmtFinType2)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# BsmtFinSF2
data_full$BsmtFinSF2[is.na(data_full$BsmtFinSF2)] <- 0

ggplot(data_full, aes(BsmtFinSF2)) +
  geom_histogram(binwidth = 100, fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# BsmtUnfSF
data_full$BsmtUnfSF[is.na(data_full$BsmtUnfSF)] <- 0

ggplot(data_full, aes(BsmtUnfSF)) +
  geom_histogram(binwidth = 250, fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# TotalBsmtSF
data_full$TotalBsmtSF[is.na(data_full$TotalBsmtSF)] <- 0

ggplot(data_full, aes(TotalBsmtSF)) +
  geom_histogram(binwidth = 250, fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# Heating
data_full$Heating <- as.factor(data_full$Heating)

ggplot(data_full, aes(Heating)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# HeatingQC
data_full$HeatingQC <- plyr::revalue(data_full$HeatingQC, c("Ex" = "5",
                                                            "Gd" = "4",
                                                            "TA" = "3",
                                                            "Fa" = "2",
                                                            "Po" = "1"))
data_full$HeatingQC <- as.numeric(data_full$HeatingQC)

data_full <- data_full %>%
  mutate(HeatingQC = ordered(HeatingQC, levels = c(1:5)))

ggplot(data_full, aes(HeatingQC)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# CentralAir
data_full$CentralAir <- as.factor(data_full$CentralAir)

ggplot(data_full, aes(CentralAir)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# Electrical
data_full$Electrical[is.na(data_full$Electrical)] <- "SBrkr"
data_full$Electrical <- ifelse(data_full$Electrical == "FuseA", "Other",
                               ifelse(data_full$Electrical == "FuseF", "Other",
                                      ifelse(data_full$Electrical == "FuseP", "Other",
                                             ifelse(data_full$Electrical == "Mix", "Other", "SBrkr"))))
data_full$Electrical <- as.factor(data_full$Electrical)

ggplot(data_full, aes(Electrical)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# FirstFloorSF & SecondFloorSF
data_full <- data_full %>%
  rename(FirstFloorSF = X1stFlrSF,
         SecondFloorSF = X2ndFlrSF)

data_full$FirstFloorSF <- as.numeric(data_full$FirstFloorSF)

ggplot(data_full, aes(FirstFloorSF)) +
  geom_histogram(binwidth = 100, fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

data_full$SecondFloorSF <- as.numeric(data_full$SecondFloorSF)

ggplot(data_full, aes(SecondFloorSF)) +
  geom_histogram(binwidth = 100, fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# LowQualFinSF
data_full$LowQualFinSF <- as.numeric(data_full$LowQualFinSF)

ggplot(data_full, aes(LowQualFinSF)) +
  geom_histogram(binwidth = 100, fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# GrLivArea
data_full$GrLivArea <- as.numeric(data_full$GrLivArea)

ggplot(data_full, aes(GrLivArea)) +
  geom_histogram(binwidth = 100, fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# BsmtFullBath
data_full$BsmtFullBath[is.na(data_full$BsmtFull)] <- 0

ggplot(data_full, aes(BsmtFullBath)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# BsmtHalfBath
data_full$BsmtHalfBath[is.na(data_full$BsmtHalfBath)] <- 0

ggplot(data_full, aes(BsmtHalfBath)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# FullBath
data_full$FullBath <- as.numeric(data_full$FullBath)

ggplot(data_full, aes(FullBath)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# HalfBath
data_full$HalfBath <- as.numeric(data_full$HalfBath)

ggplot(data_full, aes(HalfBath)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# Bedroom
data_full <- data_full %>%
  rename(Bedroom = BedroomAbvGr)

data_full$Bedroom <- as.numeric(data_full$Bedroom)

ggplot(data_full, aes(Bedroom)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# Kitchen
data_full <- data_full %>%
  rename(Kitchen = KitchenAbvGr)

data_full$Kitchen <- as.numeric(data_full$Kitchen)

ggplot(data_full, aes(Kitchen)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# KitchenQual
data_full$KitchenQual <- as.factor(data_full$KitchenQual)
data_full$KitchenQual[is.na(data_full$KitchenQual)] <- "TA"
data_full$KitchenQual <- plyr::revalue(data_full$KitchenQual, c("Fa" = "1",
                                                                "TA" = "2",
                                                                "Gd" = "3",     
                                                                "Ex" = "4"))
data_full$KitchenQual <- as.numeric(data_full$KitchenQual)

ggplot(data_full, aes(KitchenQual)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# TotRmsAbvGrd
data_full$TotRmsAbvGrd <- as.numeric(data_full$TotRmsAbvGrd)

ggplot(data_full, aes(TotRmsAbvGrd)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# Functional
data_full$Functional[is.na(data_full$Functional)] <- "Typ"
data_full$Functional <- as.factor(data_full$Functional)

ggplot(data_full, aes(Functional)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# Fireplaces
data_full$Fireplaces <- as.numeric(data_full$Fireplaces)

ggplot(data_full, aes(Fireplaces)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# FireplaceQu
data_full$FireplaceQu[is.na(data_full$FireplaceQu)] <- "0" # "None"
data_full$FireplaceQu <- plyr::revalue(data_full$FireplaceQu, c("Ex" = "6",
                                                                "Gd" = "5",
                                                                "TA" = "4",
                                                                "Fa" = "3",
                                                                "Po" = "2"))
data_full$FireplaceQu <- as.numeric(data_full$FireplaceQu)

ggplot(data_full, aes(FireplaceQu)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# GarageType
data_full$GarageType[is.na(data_full$GarageType)] <- "None"
data_full$GarageType <- as.factor(data_full$GarageType)

ggplot(data_full, aes(GarageType)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

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

ggplot(data_full, aes(GarageDecadeBuilt)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# GarageFinish
data_full$GarageFinish[is.na(data_full$GarageFinish)] <- "None"
data_full$GarageFinish <- as.factor(data_full$GarageFinish)

ggplot(data_full, aes(GarageFinish)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# GarageCars
data_full$GarageCars[is.na(data_full$GarageCars)] <- 0

ggplot(data_full, aes(GarageCars)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# GarageArea
data_full$GarageArea[is.na(data_full$GarageArea)] <- 0

ggplot(data_full, aes(GarageArea)) +
  geom_histogram(binwidth = 100, fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# GarageQual
data_full$GarageQual[is.na(data_full$GarageQual)] <- "0" # "None"
data_full$GarageQual <- plyr::revalue(data_full$GarageQual, c("Ex" = "6",
                                                              "Gd" = "5",
                                                              "TA" = "4",
                                                              "Fa" = "3",
                                                              "Po" = "2"))
data_full$GarageQual <- as.numeric(data_full$GarageQual)

ggplot(data_full, aes(GarageQual)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# GarageCond
data_full$GarageCond[is.na(data_full$GarageCond)] <- "0" # "None"
data_full$GarageCond <- plyr::revalue(data_full$GarageCond, c("Ex" = "6",
                                                              "Gd" = "5",
                                                              "TA" = "4",
                                                              "Fa" = "3",
                                                              "Po" = "2"))
data_full$GarageCond <- as.numeric(data_full$GarageCond)

ggplot(data_full, aes(GarageCond)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# PavedDrive
data_full$PavedDrive <- as.factor(data_full$PavedDrive)

ggplot(data_full, aes(PavedDrive)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# WoodDeckSF
data_full$WoodDeckSF <- as.numeric(data_full$WoodDeckSF)

ggplot(data_full, aes(WoodDeckSF)) +
  geom_histogram(binwidth = 100, fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# OpenPorchSF
data_full$OpenPorchSF <- as.numeric(data_full$OpenPorchSF)

ggplot(data_full, aes(OpenPorchSF)) +
  geom_histogram(binwidth = 100, fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# EncolsedPorch
data_full$EnclosedPorch <- as.numeric(data_full$EnclosedPorch)

ggplot(data_full, aes(EnclosedPorch)) +
  geom_histogram(binwidth = 100, fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# ThreeSeasonPorchSF
data_full <- data_full %>%
  rename(ThreeSeasonPorch = X3SsnPorch)
data_full$ThreeSeasonPorch <- as.numeric(data_full$ThreeSeasonPorch)

ggplot(data_full, aes(ThreeSeasonPorch)) +
  geom_histogram(binwidth = 100, fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# ScreenPorch
data_full$ScreenPorch <- as.numeric(data_full$ScreenPorch)

ggplot(data_full, aes(ScreenPorch)) +
  geom_histogram(binwidth = 100, fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# PoolArea 
data_full$PoolArea <- as.numeric(data_full$PoolArea)

ggplot(data_full, aes(PoolArea)) +
  geom_histogram(binwidth = 100, fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# PoolQC
data_full$PoolQC <- as.factor(data_full$PoolQC)

ggplot(data_full, aes(PoolQC)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# Fence
data_full$Fence[is.na(data_full$Fence)] <- "None"
data_full$Fence <- as.factor(data_full$Fence)

ggplot(data_full, aes(Fence)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# MiscFeature
data_full$MiscFeature[is.na(data_full$MiscFeature)] <- "None"
data_full$MiscFeature <- as.factor(data_full$MiscFeature)

ggplot(data_full, aes(MiscFeature)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# MiscVal
data_full$MiscVal <- as.numeric(data_full$MiscVal)

ggplot(data_full, aes(MiscVal)) +
  geom_histogram(binwidth = 250, fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# MoSold
data_full$MoSold <- as.factor(data_full$MoSold)

ggplot(data_full, aes(MoSold)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# YrSold 
data_full$YrSold <- as.numeric(data_full$YrSold)

ggplot(data_full, aes(YrSold)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# SaleType
data_full$SaleType[is.na(data_full$SaleType)] <- "WD"
data_full$SaleType <- as.factor(data_full$SaleType)

ggplot(data_full, aes(SaleType)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# SaleCondition
data_full$SaleCondition <- plyr::revalue(data_full$SaleCondition, c("AdjLand" = "Combine",
                                                                    "Alloca" = "Combine",
                                                                    "Family" = "Combine"))
data_full$SaleCondition <- as.factor(data_full$SaleCondition)

ggplot(data_full, aes(SaleCondition)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme


#------------------- FEATURE ENGINEERING -------------------#

# houseAge
data_full <- data_full %>%
  mutate(houseAge = (YrSold + 1) - YearBuilt)

ggplot(data_full, aes(houseAge)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

ggplot(data_full, aes(group = houseAge, houseAge, SalePrice)) +
  geom_boxplot(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

# newHouse
data_full$newHouse <- ifelse(data_full$houseAge < 3, 1, 0)

ggplot(data_full, aes(newHouse)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

ggplot(data_full, aes(group = newHouse, newHouse, SalePrice)) +
  geom_boxplot(fill = "#2c3539", col = "#a3c1ad", size = 1.25) +
  scale_y_continuous(labels = scales::dollar) +
  ylab("SalePrice") +
  plot_theme

# StructureSF, OpenPorchSF, FeatureCount, TotalRooms, TotalBathrooms, avgRoomSize, PorchSF
data_full <- data_full %>%
  mutate(PorchSF = WoodDeckSF + OpenPorchSF + EnclosedPorch + ThreeSeasonPorch + ScreenPorch,
         StructureSF = TotalBsmtSF + GrLivArea + GarageArea + PorchSF + LowQualFinSF,
         TotalBathrooms = BsmtFullBath + BsmtHalfBath + FullBath + HalfBath,
         extraRooms = (TotRmsAbvGrd - Bedroom),
         TotalRooms = (TotalBathrooms + Bedroom + Kitchen) + extraRooms,
         FeatureCount = TotalRooms + Fireplaces + GarageCars + TotalBathrooms,
         avgRoomSize = StructureSF / TotalRooms,
         sizeRatio = StructureSF / LotArea,
         sizeRatio2 = LotArea / StructureSF,
         avgFeatureSize = StructureSF / FeatureCount,
         avgBathCount = StructureSF / TotalBathrooms,
         yardSize = LotArea - (FirstFloorSF + GarageArea + PoolArea))

str(data_full[, c("PorchSF", "StructureSF", "TotalBathrooms", "extraRooms", 
                  "TotalRooms", "FeatureCount", "avgRoomSize", "sizeRatio", 
                  "sizeRatio2", "avgFeatureSize", "avgBathCount", "yardSize")])

plot_histogram(data_full[, c("PorchSF", "StructureSF", "TotalBathrooms", "extraRooms", 
                             "TotalRooms", "FeatureCount", "avgRoomSize", "sizeRatio", 
                             "sizeRatio2", "avgFeatureSize", "avgBathCount", "yardSize")],
               theme_config = list("rect" = element_blank(),
                                   "text" = element_text(colour = "#2c3539", size = 12),
                                   "line" = element_blank(),
                                   "axis.text.x" = element_text(angle = 45, hjust = 1,
                                                                colour = "#2c3539", size = 12),
                                   "axis.text.y" = element_text(colour = "#2c3539", size = 12),
                                   "axis.line.y" = element_line(colour = "#2c3539", size = .7),
                                   "axis.line.x" = element_line(colour = "#2c3539", size = .7),
                                   "axis.title" = element_text(colour = "#2c3539", size = 12)))

# medianNeighborhoodPriceSF
median_priceSF <- data_full %>%
  filter(dataPartition == "Train") %>%
  select(Neighborhood, StructureSF, SalePrice) %>%
  mutate(priceSF = SalePrice / StructureSF) %>%
  group_by(Neighborhood) %>%
  summarise(medianpriceSF = median(priceSF)) %>%
  mutate(medianNeighborhoodPriceSF = cut(medianpriceSF, breaks = 10, labels = c(1:10))) %>%
  select(Neighborhood, medianNeighborhoodPriceSF)

data_full <- left_join(data_full, median_priceSF, by.x = "Neighborhood", by.y = "Neighborhood")
data_full$medianNeighborhoodPriceSF <- as.numeric(data_full$medianNeighborhoodPriceSF)

ggplot(data_full, aes(medianNeighborhoodPriceSF)) +
  geom_bar(fill = "#C0C0C0", col = "#2c3539") +
  ylab("Count") +
  plot_theme

ggplot(data_full, aes(group = medianNeighborhoodPriceSF, medianNeighborhoodPriceSF, SalePrice)) +
  geom_boxplot(fill = "#C0C0C0", col = "#2c3539") +
  scale_y_continuous(labels = scales::dollar) +
  ylab("SalePrice") +
  plot_theme

data_full %>%
  filter(dataPartition == "Train") %>%
  ggplot() +
  geom_boxplot(aes(group = medianNeighborhoodPriceSF, medianNeighborhoodPriceSF, SalePrice),
               fill = "#a3c1ad",
               col = "#002147",
               size = 2,
               alpha = .5) +
  ylab("SalePrice") +
  xlab("medianNeighborhoodPriceSF") +
  ggtitle("Boxplot plot of medianNeighborhoodPriceSF ~ SalePrice") +
  scale_y_continuous(labels = scales::dollar) +
  plot_theme

data_full %>%
  filter(dataPartition == "Train") %>%
  mutate(priceSF = SalePrice / StructureSF) %>%
  ggplot() +
  geom_point(aes(priceSF, SalePrice),
             fill = "#a3c1ad",
             col = "#002147",
             size = 2,
             alpha = .5) +
  stat_smooth(aes(priceSF, SalePrice),
              col = "#800020",
              method = "lm",
              size = 2,
              alpha = .5,
              se = F) +
  ylab("SalePrice") +
  xlab("priceSF") +
  scale_y_continuous(labels = scales::dollar) +
  plot_theme

# medianNeighborhoodSalePrice
median_neighborhood_value <- data_full %>%
  filter(dataPartition == "Train") %>%
  select(Neighborhood, SalePrice) %>%
  group_by(Neighborhood) %>%
  summarise(medianSalePrice = median(SalePrice)) %>%
  mutate(medianNeighborhoodSalePrice = cut(medianSalePrice, breaks = 10, labels = c(1:10))) %>%
  select(Neighborhood, medianNeighborhoodSalePrice)

data_full <- left_join(data_full, median_neighborhood_value, by.x = "Neighborhood", by.y = "Neighborhood")
data_full$medianNeighborhoodSalePrice <- as.numeric(data_full$medianNeighborhoodSalePrice)

ggplot(data_full, aes(medianNeighborhoodSalePrice)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

data_full %>%
  filter(dataPartition == "Train") %>%
  ggplot() +
  geom_boxplot(aes(group = medianNeighborhoodSalePrice, medianNeighborhoodSalePrice, SalePrice),
               fill = "#a3c1ad",
               col = "#002147",
               size = 2,
               alpha = .5) +
  ylab("SalePrice") +
  xlab("medianNeighborhoodSalePrice") +
  scale_y_continuous(labels = scales::dollar) +
  plot_theme

data_full %>%
  filter(dataPartition == "Train") %>%
  ggplot() +
  geom_boxplot(aes(Neighborhood, SalePrice),
               fill = "#a3c1ad",
               col = "#002147",
               size = 2,
               alpha = .5) +
  ylab("SalePrice") +
  xlab("Neighborhood") +
  ggtitle("Boxplot of Neighborhood ~ SalePrice") +
  scale_y_continuous(labels = scales::dollar) +
  plot_theme
  
  
# medianNeighborhoodExtraRooms 
median_extraRooms <- data_full %>%
  select(Neighborhood, extraRooms) %>%
  group_by(Neighborhood) %>%
  summarise(medianextraRooms = median(extraRooms)) %>%
  mutate(medianNeighborhoodExtraRooms = cut(medianextraRooms, breaks = 10, labels = c(1:10))) %>%
  select(Neighborhood, medianNeighborhoodExtraRooms)

data_full <- left_join(data_full, median_extraRooms, by.x = "Neighborhood", by.y = "Neighborhood")
data_full$medianNeighborhoodExtraRooms <- as.numeric(data_full$medianNeighborhoodExtraRooms)

ggplot(data_full, aes(medianNeighborhoodExtraRooms)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

ggplot(data_full, aes(group = medianNeighborhoodExtraRooms, medianNeighborhoodExtraRooms, SalePrice)) +
  geom_boxplot(fill = "#2c3539", col = "#C0C0C0") +
  scale_y_continuous(labels = scales::dollar) +
  ylab("SalePrice") +
  plot_theme

# medianNeighborhoodBathrooms
median_Bathrooms <- data_full %>%
  select(Neighborhood, TotalBathrooms) %>%
  group_by(Neighborhood) %>%
  summarise(medianBathrooms = median(TotalBathrooms)) %>%
  mutate(medianNeighborhoodBathrooms = cut(medianBathrooms, breaks = 10, labels = c(1:10))) %>%
  select(Neighborhood, medianNeighborhoodBathrooms)

data_full <- left_join(data_full, median_Bathrooms, by.x = "Neighborhood", by.y = "Neighborhood") 
data_full$medianNeighborhoodBathrooms <- as.numeric(data_full$medianNeighborhoodBathrooms)

ggplot(data_full, aes(medianNeighborhoodBathrooms)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

ggplot(data_full, aes(group = medianNeighborhoodBathrooms, medianNeighborhoodBathrooms, SalePrice)) +
  geom_boxplot(fill = "#2c3539", col = "#C0C0C0") +
  scale_y_continuous(labels = scales::dollar) +
  ylab("SalePrice") +
  plot_theme

# medianNeighborhoodRooms
median_Rooms <- data_full %>%
  select(Neighborhood, TotalRooms) %>%
  group_by(Neighborhood) %>%
  summarise(medianRooms = median(TotalRooms)) %>%
  mutate(medianNeighborhoodRooms = cut(medianRooms, breaks = 10, labels = c(1:10))) %>%
  select(Neighborhood, medianNeighborhoodRooms)

data_full <- left_join(data_full, median_Rooms, by.x = "Neighborhood", by.y = "Neighborhood")
data_full$medianNeighborhoodRooms <- as.numeric(data_full$medianNeighborhoodRooms)

ggplot(data_full, aes(medianNeighborhoodRooms)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

ggplot(data_full, aes(group = medianNeighborhoodRooms, medianNeighborhoodRooms, SalePrice)) +
  geom_boxplot(fill = "#2c3539", col = "#C0C0C0") +
  scale_y_continuous(labels = scales::dollar) +
  ylab("SalePrice") +
  plot_theme

# medianNeighborhoodFeatureCount
median_FeatureCount <- data_full %>%
  select(Neighborhood, FeatureCount) %>%
  group_by(Neighborhood) %>%
  summarise(medianFeatureCount = median(FeatureCount)) %>%
  mutate(medianNeighborhoodFeatureCount = cut(medianFeatureCount, breaks = 10, labels = c(1:10))) %>%
  select(Neighborhood, medianNeighborhoodFeatureCount)

data_full <- left_join(data_full, median_FeatureCount, by.x = "Neighborhood", by.y = "Neighborhood")
data_full$medianNeighborhoodFeatureCount <- as.numeric(data_full$medianNeighborhoodFeatureCount)

ggplot(data_full, aes(medianNeighborhoodFeatureCount)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

ggplot(data_full, aes(group = medianNeighborhoodFeatureCount, medianNeighborhoodFeatureCount, SalePrice)) +
  geom_boxplot(fill = "#2c3539", col = "#C0C0C0") +
  scale_y_continuous(labels = scales::dollar) +
  ylab("SalePrice") +
  plot_theme

# medianNeighborhoodStructureSF
median_Structure_SF <- data_full %>%
  select(Neighborhood, StructureSF) %>%
  group_by(Neighborhood) %>%
  summarise(medianStructureSF = median(StructureSF)) %>%
  mutate(medianNeighborhoodStructureSF = cut(medianStructureSF, breaks = 10, labels = c(1:10))) %>%
  select(Neighborhood, medianNeighborhoodStructureSF)

data_full <- left_join(data_full, median_Structure_SF, by.x = "Neighborhood", by.y = "Neighborhood")
data_full$medianNeighborhoodStructureSF <- as.numeric(data_full$medianNeighborhoodStructureSF)

ggplot(data_full, aes(medianNeighborhoodStructureSF)) +
  geom_bar(fill = "#C0C0C0", col = "#2c3539", size = 1.25) +
  ylab("Count") +
  plot_theme

ggplot(data_full, aes(group = medianNeighborhoodStructureSF, medianNeighborhoodStructureSF, SalePrice)) +
  geom_boxplot(fill = "#C0C0C0", col = "#2c3539", size = 1.25) +
  scale_y_continuous(labels = scales::dollar) +
  ylab("SalePrice") +
  plot_theme

# medianNeighborhoodSizeRatio
median_SizeRatio <- data_full %>%
  select(Neighborhood, sizeRatio) %>%
  group_by(Neighborhood) %>%
  summarise(medianSizeRatio = median(sizeRatio)) %>%
  mutate(medianNeighborhoodsizeRatio = cut(medianSizeRatio, breaks = 10, labels = c(1:10))) %>%
  select(Neighborhood, medianNeighborhoodsizeRatio)

data_full <- left_join(data_full, median_SizeRatio, by.x = "Neighborhood", by.y = "Neighborhood")
data_full$medianNeighborhoodsizeRatio <- as.numeric(data_full$medianNeighborhoodsizeRatio)

ggplot(data_full, aes(medianNeighborhoodsizeRatio)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

ggplot(data_full, aes(group = medianNeighborhoodsizeRatio, medianNeighborhoodsizeRatio, SalePrice)) +
  geom_boxplot(fill = "#2c3539", col = "#C0C0C0") +
  scale_y_continuous(labels = scales::dollar) +
  ylab("SalePrice") +
  plot_theme

# medianNeighborhoodLotArea
median_LotArea <- data_full %>%
  select(Neighborhood, LotArea) %>%
  group_by(Neighborhood) %>%
  summarise(medianLotArea = median(LotArea)) %>%
  mutate(medianNeighborhoodLotArea = cut(medianLotArea, breaks = 10, labels = c(1:10))) %>%
  select(Neighborhood, medianNeighborhoodLotArea)

data_full <- left_join(data_full, median_LotArea, by.x = "Neighborhood", by.y = "Neighborhood")
data_full$medianNeighborhoodLotArea <- as.numeric(data_full$medianNeighborhoodLotArea)

ggplot(data_full, aes(medianNeighborhoodLotArea)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

ggplot(data_full, aes(group = medianNeighborhoodLotArea, medianNeighborhoodLotArea, SalePrice)) +
  geom_boxplot(fill = "#2c3539", col = "#C0C0C0") +
  scale_y_continuous(labels = scales::dollar) +
  ylab("SalePrice") +
  plot_theme

# medianNeighborhoodOverallQual
median_OverallQual <- data_full %>%
  select(Neighborhood, OverallQual) %>%
  group_by(Neighborhood) %>%
  summarise(medianOverallQual = median(OverallQual)) %>%
  mutate(medianNeighborhoodOverallQual = cut(medianOverallQual, breaks = 10, labels = c(1:10))) %>%
  select(Neighborhood, medianNeighborhoodOverallQual)

data_full <- left_join(data_full, median_OverallQual, by.x = "Neighborhood", by.y = "Neighborhood")
data_full$medianNeighborhoodOverallQual <- as.numeric(data_full$medianNeighborhoodOverallQual)

ggplot(data_full, aes(medianNeighborhoodOverallQual)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

ggplot(data_full, aes(group = medianNeighborhoodOverallQual, medianNeighborhoodOverallQual, SalePrice)) +
  geom_boxplot(fill = "#2c3539", 
               col = "#2c3539",
               size = .75,
               alpha = .5) +
  scale_y_continuous(labels = scales::dollar) +
  ylab("SalePrice") +
  plot_theme

# meanNeighborhoodOverallCond
mean_OverallCond <- data_full %>%
  select(Neighborhood, OverallCond) %>%
  group_by(Neighborhood) %>%
  summarise(meanOverallCond = mean(OverallCond)) %>%
  mutate(meanNeighborhoodOverallCond = cut(meanOverallCond, breaks = 10, labels = c(1:10))) %>%
  select(Neighborhood, meanNeighborhoodOverallCond) 

data_full <- left_join(data_full, mean_OverallCond, by.x = "Neighborhood", by.y = "Neighborhood") 
data_full$meanNeighborhoodOverallCond <- as.numeric(data_full$meanNeighborhoodOverallCond)

ggplot(data_full, aes(meanNeighborhoodOverallCond)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

ggplot(data_full, aes(group = meanNeighborhoodOverallCond, meanNeighborhoodOverallCond, SalePrice)) +
  geom_boxplot(fill = "#2c3539", col = "#C0C0C0") +
  scale_y_continuous(labels = scales::dollar) +
  ylab("SalePrice") +
  plot_theme

# medianNeighborhoodRoomSize
median_RoomSize <- data_full %>%
  select(Neighborhood, avgRoomSize) %>%
  group_by(Neighborhood) %>%
  summarise(medianRoomSize = median(avgRoomSize)) %>%
  mutate(medianNeighborhoodRoomSize = cut(medianRoomSize, breaks = 10, labels = c(1:10))) %>%
  select(Neighborhood, medianNeighborhoodRoomSize)

data_full <- left_join(data_full, median_RoomSize, by.x = "Neighborhood", by.y = "Neighborhood")
data_full$medianNeighborhoodRoomSize <- as.numeric(data_full$medianNeighborhoodRoomSize)

ggplot(data_full, aes(medianNeighborhoodRoomSize)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

ggplot(data_full, aes(group = medianNeighborhoodRoomSize, medianNeighborhoodRoomSize, SalePrice)) +
  geom_boxplot(fill = "#2c3539", col = "#C0C0C0") +
  scale_y_continuous(labels = scales::dollar) +
  ylab("SalePrice") +
  plot_theme

# medianNeighborhoodyardSize
median_yardSize <- data_full %>%
  select(Neighborhood, yardSize) %>%
  group_by(Neighborhood) %>%
  summarise(medianYardSize = median(yardSize)) %>%
  mutate(medianNeighborhoodyardSize = cut(medianYardSize, breaks = 10, labels = c(1:10))) %>%
  select(Neighborhood, medianNeighborhoodyardSize)

data_full <- left_join(data_full, median_yardSize, by.x = "Neighborhood", by.y = "Neighborhood")
data_full$medianNeighborhoodyardSize <- as.numeric(data_full$medianNeighborhoodyardSize)

ggplot(data_full, aes(medianNeighborhoodyardSize)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

ggplot(data_full, aes(group = medianNeighborhoodyardSize, medianNeighborhoodyardSize, SalePrice)) +
  geom_boxplot(fill = "#2c3539", col = "#C0C0C0") +
  scale_y_continuous(labels = scales::dollar) +
  ylab("SalePrice") +
  plot_theme

# meanNeighborhoodAge
median_Age <- data_full %>%
  select(Neighborhood, houseAge) %>%
  group_by(Neighborhood) %>%
  summarise(meanAge = mean(houseAge)) %>%
  mutate(meanNeighborhoodAge = cut(meanAge, breaks = 10, labels = c(1:10))) %>%
  select(Neighborhood, meanNeighborhoodAge)

data_full <- left_join(data_full, median_Age, by.x = "Neighborhood", by.y = "Neighborhood")
data_full$meanNeighborhoodAge <- as.numeric(data_full$meanNeighborhoodAge)

ggplot(data_full, aes(meanNeighborhoodAge)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

ggplot(data_full, aes(group = meanNeighborhoodAge, meanNeighborhoodAge, SalePrice)) +
  geom_boxplot(fill = "#2c3539", col = "#C0C0C0") +
  scale_y_continuous(labels = scales::dollar) +
  ylab("SalePrice") +
  plot_theme

# medianMSSubClassStructureSF
medianMSSubClass_StructureSF <- data_full %>%
  select(MSSubClass, StructureSF) %>%
  group_by(MSSubClass) %>%
  summarise(medianStructureSF = median(StructureSF)) %>%
  mutate(medianMSSubClassStructureSF = cut(medianStructureSF, breaks = 10, labels = c(1:10))) %>%
  select(MSSubClass, medianMSSubClassStructureSF)

data_full <- left_join(data_full, medianMSSubClass_StructureSF, by.x = "MSSubClass", by.y = "MSSubClass")
data_full$medianMSSubClassStructureSF <- as.numeric(data_full$medianMSSubClassStructureSF)

data_full %>%
  filter(dataPartition == "Train") %>%
  ggplot() +
  geom_boxplot(aes(group = medianMSSubClassStructureSF, medianMSSubClassStructureSF, SalePrice),
               fill = "#C0C0C0",
               col = "#2c3539",
               size = 1.25) +
  scale_y_continuous(labels = scales::dollar) +
  ylab("SalePrice") +
  xlab("medianMSSubClassStructureSF") +
  plot_theme

# medianMSSubClassPriceSF
medianMSSubClass_PriceSF <- data_full %>%
  filter(dataPartition == "Train") %>%
  select(MSSubClass, StructureSF, SalePrice) %>%
  mutate(priceSF = SalePrice / StructureSF) %>%
  group_by(MSSubClass) %>%
  summarise(medianPriceSF = median(priceSF)) %>%
  mutate(medianMSSubClassPriceSF = cut(medianPriceSF, breaks = 10, labels = c(1:10))) %>%
  select(MSSubClass, medianMSSubClassPriceSF)

data_full <- left_join(data_full, medianMSSubClass_PriceSF, by.x = "MSSubClass", by.y = "MSSubClass")
data_full$medianMSSubClassPriceSF <- as.numeric(data_full$medianMSSubClassPriceSF)

data_full %>%
  filter(dataPartition == "Train") %>%
  ggplot() +
  geom_boxplot(aes(group = medianMSSubClassPriceSF, medianMSSubClassPriceSF, SalePrice),
               fill = "#C0C0C0",
               col = "#2c3539",
               size = 1.25) +
  scale_y_continuous(labels = scales::dollar) +
  ylab("SalePrice") +
  xlab("medianMSSubClassPriceSF") +
  plot_theme

# medianMSSubClassSalePrice
medianMSSubClass_SalePrice <- data_full %>%
  filter(dataPartition == "Train") %>%
  select(MSSubClass, SalePrice) %>%
  group_by(MSSubClass) %>%
  summarise(medianSalePrice = median(SalePrice)) %>%
  mutate(medianMSSubClassSalePrice = cut(medianSalePrice, breaks = 10, labels = c(1:10))) %>%
  select(MSSubClass, medianMSSubClassSalePrice)

data_full <- left_join(data_full, medianMSSubClass_SalePrice, by.x = "MSSubClass", by.y = "MSSubClass")
data_full$medianMSSubClassSalePrice <- as.numeric(data_full$medianMSSubClassSalePrice)

data_full %>%
  filter(dataPartition == "Train") %>%
  ggplot() +
  geom_boxplot(aes(group = medianMSSubClassSalePrice, medianMSSubClassSalePrice, SalePrice),
               fill = "#FFFFFF",
               col = "#2c3539",
               size = 1.25,
               alpha = .5) +
  ylab("SalePrice") +
  xlab("medianMSSubClassSalePrice") +
  scale_y_continuous(labels = scales::dollar) +
  plot_theme

# medianMSSubClassFeatureCount
medianMSSubClassFeatureCount <- data_full %>%
  select(MSSubClass, FeatureCount) %>%
  group_by(MSSubClass) %>%
  summarise(medianFeatureCount = median(FeatureCount)) %>%
  mutate(medianMSSubClassFeatureCount = cut(medianFeatureCount, breaks = 10, labels = c(1:10))) %>%
  select(MSSubClass, medianMSSubClassFeatureCount)

data_full <- left_join(data_full, medianMSSubClassFeatureCount, by.x = "MSSubClass", by.y = "MSSubClass")
data_full$medianMSSubClassFeatureCount <- as.numeric(data_full$medianMSSubClassFeatureCount)

data_full %>%
  filter(dataPartition == "Train") %>%
  ggplot() +
  geom_boxplot(aes(group = medianMSSubClassFeatureCount, medianMSSubClassFeatureCount, SalePrice),
               fill = "#C0C0C0",
               col = "#2c3539",
               size = 1.25) +
  ylab("SalePrice") +
  xlab("medianMSSubClassFeatureCount") +
  scale_y_continuous(labels = scales::dollar) +
  plot_theme

# medianMSZoningSalePrice
medianMSZoning_SalePrice <- data_full %>%
  filter(dataPartition == "Train") %>%
  select(MSZoning, SalePrice) %>%
  group_by(MSZoning) %>%
  summarise(medianSalePrice = median(SalePrice)) %>%
  mutate(medianMSZoningSalePrice = cut(medianSalePrice, breaks = 10, labels = c(1:10))) %>%
  select(MSZoning, medianMSZoningSalePrice)

data_full <- left_join(data_full, medianMSZoning_SalePrice, by.x = "MSZoning", by.y = "MSZoning")
data_full$medianMSZoningSalePrice <- as.numeric(data_full$medianMSZoningSalePrice)

data_full %>%
  filter(dataPartition == "Train") %>%
  ggplot() +
  geom_boxplot(aes(group = medianMSZoningSalePrice, medianMSZoningSalePrice, SalePrice),
               fill = "#a3c1ad",
               col = "#002147",
               size = 1.25,
               alpha = .5) +
  ylab("SalePrice") +
  xlab("medianMSZoningSalePrice") +
  scale_y_continuous(labels = scales::dollar) +
  plot_theme

# hasPorch
data_full$hasPorch <- ifelse(data_full$PorchSF > 0, 1, 0)
data_full$hasPorch <- as.factor(data_full$hasPorch)

ggplot(data_full, aes(hasPorch)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

ggplot(data_full, aes(group = hasPorch, hasPorch, SalePrice)) +
  geom_boxplot(fill = "#2c3539", col = "#C0C0C0") +
  scale_y_continuous(labels = scales::dollar) +
  ylab("SalePrice") +
  plot_theme

# hasGarage
data_full$hasGarage <- ifelse(data_full$GarageQual == 0, 0, 1)
data_full$hasGarage <- as.factor(data_full$hasGarage)

ggplot(data_full, aes(hasGarage)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

ggplot(data_full, aes(group = hasGarage, hasGarage, SalePrice)) +
  geom_boxplot(fill = "#2c3539", col = "#C0C0C0") +
  scale_y_continuous(labels = scales::dollar) +
  ylab("SalePrice") +
  plot_theme

# hasPool
data_full$hasPool <- ifelse(data_full$PoolArea == 0, 0, 1)
data_full$hasPool <- as.factor(data_full$hasPool)

ggplot(data_full, aes(hasPool)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

ggplot(data_full, aes(group = hasPool, hasPool, SalePrice)) +
  geom_boxplot(fill = "#2c3539", col = "#C0C0C0") +
  scale_y_continuous(labels = scales::dollar) +
  ylab("SalePrice") +
  plot_theme

# hasFireplace
data_full$hasFireplace <- ifelse(data_full$Fireplaces > 0, 1, 0)
data_full$hasFireplace <- as.factor(data_full$hasFireplace)

ggplot(data_full, aes(hasFireplace)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

ggplot(data_full, aes(group = hasFireplace, hasFireplace, SalePrice)) +
  geom_boxplot(fill = "#2c3539", col = "#C0C0C0") +
  scale_y_continuous(labels = scales::dollar) +
  ylab("SalePrice") +
  plot_theme

# hasBasement
data_full$hasBasement <- ifelse(data_full$TotalBsmtSF == 0, 0, 1)
data_full$hasBasement <- as.factor(data_full$hasBasement)

ggplot(data_full, aes(hasBasement)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

ggplot(data_full, aes(group = hasBasement, hasBasement, SalePrice)) +
  geom_boxplot(fill = "#2c3539", col = "#C0C0C0") +
  scale_y_continuous(labels = scales::dollar) +
  ylab("SalePrice") +
  plot_theme

# Renovated
data_full$Renovated <- ifelse(data_full$YearRemodAdd == data_full$YearBuilt, 0, 1)
data_full$Renovated <- as.factor(data_full$Renovated)

ggplot(data_full, aes(Renovated)) +
  geom_bar(fill = "#2c3539", col = "#C0C0C0") +
  ylab("Count") +
  plot_theme

ggplot(data_full, aes(group = Renovated, Renovated, SalePrice)) +
  geom_boxplot(fill = "#2c3539", col = "#C0C0C0") +
  scale_y_continuous(labels = scales::dollar) +
  ylab("SalePrice") +
  plot_theme

# Convert variables to factor
data_full$Utilities <- as.factor(data_full$Utilities)
data_full$Neighborhood <- as.factor(data_full$Neighborhood)
data_full$YrSold <- as.factor(data_full$YrSold)
data_full$MoSold <- as.factor(data_full$MoSold)

# Remove PoolQC & Utilities
data_full$PoolQC <- NULL
data_full$Utilities <- NULL

#------------------- DATASET BUILD VISUALIZATION -------------------#

introduce(data_full)
str(data_full[, 1:60])
str(data_full[, 61:122])

str(data_full[, c("StructureSF")])

p1 <- data_full %>%
  filter(dataPartition == "Train") %>%
  mutate(PriceSF = SalePrice / StructureSF) %>%
  ggplot() +
  geom_point(aes(PriceSF, SalePrice), 
             col = "#2c3539",
             size = 4,
             alpha = .5) +
  stat_smooth(aes(PriceSF, SalePrice), 
              method = "loess", 
              fill = "#a3c1ad",
              col = "#002147",
              size = 1,
              alpha = .5) +
  ylab("SalePrice") +
  xlab("PriceSF") +
  ggtitle("PriceSF ~ SalePrice") +
  scale_y_continuous(labels = scales::dollar) +
  plot_theme

p2 <- data_full %>%
  filter(dataPartition == "Train") %>%
  ggplot() +
  geom_boxplot(aes(group = medianNeighborhoodPriceSF, medianNeighborhoodPriceSF, SalePrice),
               fill = "#a3c1ad",
               col = "#002147",
               size = 1,
               alpha = .5) +
  ylab("SalePrice") +
  xlab("medianNeighborhoodPriceSF") +
  ggtitle("medianNeighborhoodPriceSF to SalePrice") +
  scale_y_continuous(labels = scales::dollar) +
  plot_theme
  
p3 <- data_full %>%
  filter(dataPartition == "Train") %>%
  ggplot() +
  geom_point(aes(StructureSF, SalePrice), 
             col = "#2c3539", 
             size = 2,
             alpha = .5) +
  stat_smooth(aes(StructureSF, SalePrice),
              method = "loess",
              fill = "#a3c1ad",
              col = "#002147",
              size = 1,
              alpha = .5) +
  ylab("SalePrice") +
  xlab("StructureSF") +
  ggtitle("StructureSF ~ SalePrice") +
  scale_y_continuous(labels = scales::dollar) +
  plot_theme

p4 <- data_full %>%
  filter(dataPartition == "Train") %>%
  ggplot() +
  geom_boxplot(aes(group = medianNeighborhoodStructureSF, medianNeighborhoodStructureSF, SalePrice),
               fill = "#a3c1ad",
               col = "#002147",
               size = 1,
               alpha = .5) +
  ylab("SalePrice") +
  xlab("medianNeighborhoodStructureSF") +
  ggtitle("medianNeighborhoodStructureSF to SalePrice") +
  scale_y_continuous(labels = scales::dollar) +
  plot_theme

p5 <- data_full %>%
  ggplot() +
  geom_bar(aes(MSSubClass),
           fill = "#a3c1ad",
           col = "#002147",
           size = 1,
           alpha = .5) +
  ylab("Count") +
  xlab("MSSubClass") +
  plot_theme

p6 <- data_full %>%
  filter(dataPartition == "Train") %>%
  ggplot() +
  geom_boxplot(aes(group = MSSubClass, MSSubClass, SalePrice), 
               fill = "#a3c1ad",
               col = "#002147",
               size = 1,
               alpha = .5) +
  ylab("SalePrice") +
  xlab("MSSubClass") +
  scale_y_continuous(labels = scales::dollar) +
  plot_theme

p7 <- data_full %>%
  filter(dataPartition == "Train") %>%
  ggplot() +
  geom_histogram(aes(SalePrice),
                 binwidth = 10000,
                 fill = "#a3c1ad",
                 col = "#002147",
                 size = 1,
                 alpha = .5) +
  scale_x_continuous(labels = scales::dollar) +
  ylab("Count") +
  xlab("SalePrice") +
  plot_theme

p8 <- data_full %>%
  ggplot() +
  geom_histogram(aes(StructureSF),
                 binwidth =200,
                 fill = "#a3c1ad",
                 col = "#002147",
                 size = 1,
                 alpha = .5) +
  ylab("Count") +
  xlab("StructureSF") +
  plot_theme

p9 <- data_full %>%
  ggplot() +
  geom_bar(aes(Neighborhood),
           fill = "#a3c1ad",
           col = "#002147",
           size = 1,
           alpha = .5) +
  ylab("Count") +
  xlab("Neighborhood") +
  plot_theme

p10 <- data_full %>%
  filter(dataPartition == "Train") %>%
  ggplot() +
  geom_boxplot(aes(Neighborhood, SalePrice),
               fill = "#a3c1ad",
               col = "#002147",
               size = 1,
               alpha = .5) +
  ylab("SalePrice") +
  xlab("Neighborhood") +
  scale_y_continuous(labels = scales::dollar) +
  plot_theme +
  theme(axis.text.x = element_text(hjust = 1, angle = 45))

p11 <- data_full %>%
  ggplot() +
  geom_bar(aes(OverallQual),
           fill = "#a3c1ad",
           col = "#002147",
           size = 2,
           alpha = .5) +
  ylab("Count") +
  xlab("OverallQual") +
  plot_theme

p12 <- data_full %>%
  filter(dataPartition == "Train") %>%
  ggplot() +
  geom_boxplot(aes(group = OverallQual, OverallQual, SalePrice),
               fill = "#a3c1ad",
               col = "#002147",
               size = 1,
               alpha = .5) +
  ylab("SalePrice") +
  xlab("OverallQual") +
  scale_y_continuous(labels = scales::dollar) +
  plot_theme

p13 <- data_full %>%
  ggplot() +
  geom_bar(aes(DecadeBuilt),
           fill = "#a3c1ad",
           col = "#2c3539",
           size = 2,
           alpha = .5) +
  ylab("Count") +
  xlab("DecadeBuilt") +
  plot_theme

p14 <- data_full %>%
  filter(dataPartition == "Train") %>%
  ggplot() +
  geom_boxplot(aes(group = DecadeBuilt, DecadeBuilt, SalePrice),
               fill = "#a3c1ad",
               col = "#2c3539",
               size = .75,
               alpha = .5) +
  ylab("SalePrice") +
  xlab("DecadeBuilt") +
  scale_y_continuous(labels = scales::dollar) +
  plot_theme

grid.arrange(p1, p2, p3, 
             p4, p5, p6, 
             p7, p8, p9,
             p10, p11, p12,
             p13, p14,
             ncol = 4)

data_full %>%
  filter(dataPartition == "Train") %>%
  ggplot() +
  geom_point(aes(StructureSF, SalePrice),
             col = "#2c3539",
             size = 5,
             alpha = .5) +
  stat_smooth(aes(StructureSF, SalePrice),
              method = "lm",
              col = "#800020",
              size = 2,
              se = F) +
  ylab("SalePrice") +
  xlab("StructureSF") +
  scale_y_continuous(labels = scales::dollar) +
  plot_theme

#------------------- LINEAR MODEL(S) -------------------#

lmTrain <- data_full %>%
  filter(dataPartition == "Train") %>%
  select(StructureSF, SalePrice) %>%
  mutate(PriceSF = SalePrice / StructureSF)

# StructureSF
lm_fit_1 <- lm(SalePrice ~ StructureSF, data = lmTrain)
summary(lm_fit_1)
cor(lmTrain$SalePrice, lmTrain$StructureSF)

lmTrain$predicted <- predict(lm_fit_1)
lmTrain$residuals <- residuals(lm_fit_1)

lmTrain %>%
  select(SalePrice, predicted, residuals) %>%
  head()

p1 <- ggplot(lmTrain, aes(StructureSF, SalePrice)) +
  geom_segment(aes(xend = StructureSF, yend = predicted),
               col = "#a3c1ad",
               size = 2,
               alpha = .5) +
  stat_smooth(method = "lm", 
              se = F, 
              col = "#800020",
              size = 1.5) +
  geom_point(col = "#2c3539",
             size = 2,
             shape = 1) +
  ylab("SalePrice") +
  xlab("StructureSF") +
  ggtitle("Linear regression of SalePrice ~ StructureSF with residuals") +
  scale_y_continuous(labels = scales::dollar) +
  plot_theme

p2 <- ggplot(lmTrain, aes(residuals)) +
  geom_histogram(binwidth = 10000,
                 fill = "#a3c1ad",
                 col = "#2c3539",
                 size = .75,    
                 alpha = .2) +
  ylab("Count") +
  xlab("Residuals") +
  ggtitle("Residuals of SalePrice ~ StructureSF linear regression") +
  plot_theme

p3 <- ggplot(lmTrain, aes(SalePrice)) +
  geom_histogram(binwidth = 10000,
                 fill = "#a3c1ad",
                 col = "#2c3539",
                 size = .75,
                 alpha = .2) +
  ylab("Count") +
  xlab("SalePrice") +
  scale_x_continuous(labels = scales::dollar) +
  ggtitle("Train - SalePrice") +
  plot_theme

p4 <- ggplot(lmTrain, aes(predicted)) +
  geom_histogram(binwidth = 10000,
                 fill = "#a3c1ad",
                 col = "#2c3539",
                 size = .75,
                 alpha = .2) +
  ylab("Count") +
  xlab("Predicted values") +
  scale_x_continuous(labels = scales::dollar) +
  ggtitle("Predicted SalePrice ~ StructureSF linear regression") +
  plot_theme



lmTrain2 <- data_full %>%
  filter(dataPartition == "Train") %>%
  select(StructureSF, SalePrice) %>%
  mutate(PriceSF = SalePrice / StructureSF)

# PriceSF
lm_fit_2 <- lm(SalePrice ~ PriceSF, data = lmTrain)
summary(lm_fit_2)

lmTrain2$predicted <- predict(lm_fit_2)
lmTrain2$residuals <- residuals(lm_fit_2)

lmTrain2 %>%
  select(SalePrice, predicted, residuals) %>%
  head()

p5 <- ggplot(lmTrain2, aes(PriceSF, SalePrice)) +
  geom_segment(aes(xend = PriceSF, yend = predicted),
               col = "#a3c1ad",
               size = 2,
               alpha = .5) +
  stat_smooth(method = "lm", 
              se = F, 
              col = "#800020",
              size = 1.5) +
  geom_point(col = "#2c3539",
             size = 2,
             shape = 1) +
  ylab("SalePrice") +
  xlab("PriceSF") +
  ggtitle("Linear regression of SalePrice ~ PriceSF with residuals") +
  scale_y_continuous(labels = scales::dollar) +
  plot_theme
p6 <- ggplot(lmTrain2, aes(residuals)) +
  geom_histogram(binwidth = 10000,
                 fill = "#a3c1ad",
                 col = "#002147",
                 size = .75,    
                 alpha = .2) +
  ylab("Count") +
  xlab("Residuals") +
  ggtitle("Residuals of SalePrice ~ PriceSF linear regression") +
  plot_theme

grid.arrange(p1, p2, p3, p4, p5, p6, ncol = 2)

#------------------- WRITE AND LOAD CLEANED DATASET -------------------#

# Write csv
write.csv(data_full, file = "data_full.csv")

# Load csv
data_full <- read.csv("data_full.csv",
                      header = T,
                      stringsAsFactors = T)
data_full$X <- NULL

#------------------- DUMMY VARIABLES AND NEAR ZERO VARIANCE FEATURES -------------------#

# Pull out Id and dataPartition
dataIndexVars <- data_full %>%
  dplyr::select(Id, dataPartition, SalePrice)

data_full <- data_full %>%
  dplyr::select(-c(Id, dataPartition, SalePrice))

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
  dplyr::select(-c(Id, dataPartition))

rfe_Control <- rfeControl(functions = rfFuncs, 
                          method = "repeatedcv", 
                          number = 5,
                          repeats = 5)

rfeTime1 <- Sys.time()

set.seed(1)
rfeResults <- rfe(rfeTrain[, 2:162],
                  rfeTrain$SalePrice,
                  rfeControl = rfe_Control,
                  metric = "RMSE",
                  maximize = F,
                  preProcess = c("center", "scale"))

rfeResults
rfeTime2 <- Sys.time()
rfeElapsed <- rfeTime2 - rfeTime1
rfeElapsed

rfeVariables <- predictors(rfeResults)

model_data <- model_data[, rfeVariables]

model_data <- bind_cols(dataIndexVars, model_data)

#------------------- SPLIT INTO TRAIN & TEST -------------------#

train <- model_data %>%
  filter(dataPartition == "Train") %>%
  dplyr::select(-c(Id, dataPartition))

plot_correlation(train)

test <- model_data %>%
  filter(dataPartition == "Test") %>%
  dplyr::select(-c(dataPartition))

test.pred <- test %>%
  dplyr::select(-c(Id))

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

RF.grid <- expand.grid(mtry = 4)

XGB.grid <- expand.grid(nrounds = 3000,
                        max_depth = 8,
                        eta = 0.01,
                        gamma = 0,
                        colsample_bytree = .7,
                        min_child_weight = 1,
                        subsample = .5)

#------------------- MODELS ------------------#

ensembleTime1 <- Sys.time()

time1 <- Sys.time()

set.seed(5)
XGB.model <- train(SalePrice ~.,
                   data = train,
                   trControl = objControl,
                   metric = "RMSE",
                   method = "xgbTree",
                   tuneGrid = XGB.grid,
                   preProc = c("center", "scale"))

time2 <- Sys.time()
model1Time <- time2 - time1

time1 <- Sys.time()

set.seed(5)
GBM.model <- train(SalePrice ~ ., 
                   data = train,
                   trControl = objControl,
                   metric = "RMSE",
                   method = "gbm",
                   tuneGrid = GBM.grid,
                   preProc = c("center", "scale"))

time2 <- Sys.time()
model2Time <- time2 - time1

time1 <- Sys.time()

set.seed(6)                   
RF.model <- train(SalePrice ~ ., 
                  data = train,
                  metric = "RMSE",
                  tuneGrid = RF.grid,
                  trControl = objControl,
                  ntree = 1500,
                  preProc = c("center", "scale"))

time2 <- Sys.time()
model3Time <- time2 - time1

time1 <- Sys.time()

set.seed(7)                   
SVM.model <- train(SalePrice ~ ., 
                   data = train,
                   trControl = objControl,
                   metric = "RMSE",
                   method = "svmRadial",
                   preProc = c("center", "scale"))

time2 <- Sys.time()
model4Time <- time2 - time1

XGB.model
GBM.model
RF.model
SVM.model

ensembleTime2 <- Sys.time()
ensembleTime <- ensembleTime2 - ensembleTime1
ensembleTime

#------------------- MODEL(S) EVALUATION -------------------#

combined_models<- list(xgbTree = XGB.model,
                       gbm = GBM.model, 
                       rf = RF.model,
                       svmRadial = SVM.model)

class(combined_models) <- "caretList"

modelCor(resamples(combined_models))
summary(resamples(combined_models))

#------------------- WEIGHTED ENSEMBLE -------------------#

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

#------------------- PREDICTION & SUBMISSION -------------------#

# Numeric centered & scaled (Ensemble - GBM, RF, SVM)
preds <- predict(models_ensemble, newdata = test.pred)

ensemble_data_full <- data.frame(Id = test$Id, SalePrice = preds)
head(ensemble_data_full)

write.csv(ensemble_data_full, file = "ensemble_data_full.csv")

modelTime2 <- Sys.time()
modelElapsed <- modelTime2 - modelTime1

model1Time
model2Time
model3Time
model4Time
ensembleTime

#------------------- DATA VISUALIZATIONS -------------------#

train_SalePrice <- ggplot(train, aes(SalePrice)) +
  geom_histogram(binwidth = 10000, fill = "#002147", col = "#2c3539", alpha = .5) +
  xlab("Train SalePrice") +
  ylab("Count") +
  scale_x_continuous(labels = scales::dollar) +
  theme(rect = element_blank(),
        text = element_text(colour = "#2c3539", size = 14),
        line = element_blank(),
        axis.text.x = element_text(hjust = 1, colour = "#2c3539", size = 14),
        axis.text.y = element_text(colour = "#2c3539", size = 14),
        axis.line.y = element_line(colour = "#2c3539"),
        axis.line.x = element_line(colour = "#2c3539"),
        axis.title = element_text(colour = "#2c3539", size = 14))
pred_SalePrice <- ggplot(ensemble_data_full, aes(SalePrice)) +
  geom_histogram(binwidth = 10000, fill = "#002147", col = "#2c3539", alpha = .5) +
  xlab("Predicted SalePrice") + 
  ylab("Count") +
  scale_x_continuous(labels = scales::dollar) +
  theme(rect = element_blank(),
        text = element_text(colour = "#2c3539", size = 14),
        line = element_blank(),
        axis.text.x = element_text(hjust = 1, colour = "#2c3539", size = 14),
        axis.text.y = element_text(colour = "#2c3539", size = 14),
        axis.line.y = element_line(colour = "#2c3539"),
        axis.line.x = element_line(colour = "#2c3539"),
        axis.title = element_text(colour = "#2c3539", size = 14))
grid.arrange(train_SalePrice, pred_SalePrice, ncol = 1)

# Key Variables

keyVar_1 <- ggplot(train, aes(StructureSF, SalePrice)) +
  geom_point(col = "#2c3539") +
  ylab("SalePrice") +
  xlab("StructureSF") +
  scale_y_continuous(labels = scales::dollar) +
  theme(rect = element_blank(),
        text = element_text(colour = "#2c3539", size = 14),
        line = element_blank(),
        axis.text.x = element_text(hjust = 1, colour = "#2c3539", size = 14),
        axis.text.y = element_text(colour = "#2c3539", size = 14),
        axis.line.y = element_line(colour = "#2c3539"),
        axis.line.x = element_line(colour = "#2c3539"),
        axis.title = element_text(colour = "#2c3539", size = 14))
keyVar_2 <- ggplot(train, aes(GrLivArea, SalePrice)) +
  geom_point(col = "#2c3539") +
  ylab("SalePrice") +
  xlab("GrLivArea") +
  scale_y_continuous(labels = scales::dollar) +
  theme(rect = element_blank(),
        text = element_text(colour = "#2c3539", size = 14),
        line = element_blank(),
        axis.text.x = element_text(hjust = 1, colour = "#2c3539", size = 14),
        axis.text.y = element_text(colour = "#2c3539", size = 14),
        axis.line.y = element_line(colour = "#2c3539"),
        axis.line.x = element_line(colour = "#2c3539"),
        axis.title = element_text(colour = "#2c3539", size = 14))
keyVar_3 <- ggplot(train, aes(group = OverallQual, OverallQual, SalePrice)) +
  geom_boxplot(col = "#002147", fill = "#A9A9A9") +
  ylab("SalePrice") +
  xlab("OverallQual") +
  scale_y_continuous(labels = scales::dollar) +
  theme(rect = element_blank(),
        text = element_text(colour = "#2c3539", size = 14),
        line = element_blank(),
        axis.text.x = element_text(hjust = 1, colour = "#2c3539", size = 14),
        axis.text.y = element_text(colour = "#2c3539", size = 14),
        axis.line.y = element_line(colour = "#2c3539"),
        axis.line.x = element_line(colour = "#2c3539"),
        axis.title = element_text(colour = "#2c3539", size = 14))
keyVar_4 <- ggplot(train, aes(group = FeatureCount, FeatureCount, SalePrice)) +
  geom_boxplot(col = "#002147", fill = "#A9A9A9") +
  ylab("SalePrice") +
  xlab("FeatureCount") +
  scale_y_continuous(labels = scales::dollar) +
  theme(rect = element_blank(),
        text = element_text(colour = "#2c3539", size = 14),
        line = element_blank(),
        axis.text.x = element_text(hjust = 1, colour = "#2c3539", size = 14),
        axis.text.y = element_text(colour = "#2c3539", size = 14),
        axis.line.y = element_line(colour = "#2c3539"),
        axis.line.x = element_line(colour = "#2c3539"),
        axis.title = element_text(colour = "#2c3539", size = 14))
keyVar_5 <- ggplot(train, aes(group = neighborhoodValues, neighborhoodValues, SalePrice)) +
  geom_boxplot(col = "#002147", fill = "#A9A9A9") +
  ylab("SalePrice") +
  xlab("FeatureCount") +
  scale_y_continuous(labels = scales::dollar) +
  theme(rect = element_blank(),
        text = element_text(colour = "#2c3539", size = 14),
        line = element_blank(),
        axis.text.x = element_text(hjust = 1, colour = "#2c3539", size = 14),
        axis.text.y = element_text(colour = "#2c3539", size = 14),
        axis.line.y = element_line(colour = "#2c3539"),
        axis.line.x = element_line(colour = "#2c3539"),
        axis.title = element_text(colour = "#2c3539", size = 14))
keyVar_6 <- ggplot(train, aes(group = YearRemodAdd, YearRemodAdd, SalePrice)) +
  geom_boxplot(col = "#002147", fill = "#A9A9A9") +
  ylab("SalePrice") +
  xlab("YearRemodAdd") +
  scale_y_continuous(labels = scales::dollar) +
  theme(rect = element_blank(),
        text = element_text(colour = "#2c3539", size = 14),
        line = element_blank(),
        axis.text.x = element_text(hjust = 1, colour = "#2c3539", size = 14),
        axis.text.y = element_text(colour = "#2c3539", size = 14),
        axis.line.y = element_line(colour = "#2c3539"),
        axis.line.x = element_line(colour = "#2c3539"),
        axis.title = element_text(colour = "#2c3539", size = 14))
keyVar_7 <- ggplot(train, aes(group = medianNeighborhoodFeatureCount, medianNeighborhoodFeatureCount, SalePrice)) +
  geom_boxplot(col = "#002147", fill = "#A9A9A9") +
  ylab("SalePrice") +
  xlab("medianNeighborhoodFeatureCount") +
  scale_y_continuous(labels = scales::dollar) +
  theme(rect = element_blank(),
        text = element_text(colour = "#2c3539", size = 14),
        line = element_blank(),
        axis.text.x = element_text(hjust = 1, colour = "#2c3539", size = 14),
        axis.text.y = element_text(colour = "#2c3539", size = 14),
        axis.line.y = element_line(colour = "#2c3539"),
        axis.line.x = element_line(colour = "#2c3539"),
        axis.title = element_text(colour = "#2c3539", size = 14))

grid.arrange(keyVar_1, keyVar_2, keyVar_3, keyVar_4, keyVar_5, keyVar_6, keyVar_7, ncol = 4)
