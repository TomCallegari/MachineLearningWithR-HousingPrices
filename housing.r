
# Housing Price Competition - Kaggle (Carbonati Kernel)

require(ggplot2)
require(stringr)
require(Matrix)
require(glmnet)
require(xgboost)
require(randomForest)
require(Metrics)
require(dplyr)
require(caret)
require(scales)
require(e1071)
require(corrplot)
require(Amelia)

setwd("D://Analytics/Housing Prices")
train <- read.csv("train.csv", header = TRUE, stringsAsFactors = FALSE)
test <- read.csv("test.csv", header = TRUE, stringsAsFactors = FALSE)

# Combine Datasets
df.combined <- rbind(within(train, rm("Id", "SalePrice")), within(test, rm("Id")))
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

#
# XGBoost
#

dummy.vars <- dummyVars(~., data = df.combined)
train.dummy <- predict(dummy.vars, train)







