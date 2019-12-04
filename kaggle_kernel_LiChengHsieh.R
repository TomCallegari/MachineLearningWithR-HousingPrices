#
#
#
#
# Kaggle Kernel - Housing Prices (Li Cheng Hsieh)

library(dplyr)
library(ggplot2)
library(Amelia)
library(gridExtra)
library(caret)
library(doSNOW)
library(gbm)

setwd("D://Analytics/Housing Prices")

train <- read.csv("train.csv",
                  header = T,
                  stringsAsFactors = F)

test <- read.csv("test.csv",
                 header = T,
                 stringsAsFactors = F)

SalePrice <- train$SalePrice

df <- rbind(train[,!(names(train) %in% c("SalePrice"))], test)

df$YearBuilt<-as.factor(df$YearBuilt)
df$YearRemodAdd<-as.factor(df$YearRemodAdd)
df$MSSubClass<-as.factor(df$MSSubClass)
df$OverallQual<-as.factor(df$OverallQual)
df$OverallCond<-as.factor(df$OverallCond)
df$MoSold<-as.factor(df$MoSold)
df$YrSold<-as.factor(df$YrSold)
df$GarageYrBlt<-as.factor(df$GarageYrBlt)

str(df)

options(repr.plot.width=6, repr.plot.height=5)
cMiss = function(x){sum(is.na(x))}
CM <- sort(apply(df,2,cMiss),decreasing=T);
barplot(CM[CM!=0],
        las=2,
        cex.names=0.6,
        ylab="Count",
        ylim=c(0,3000),
        horiz=F,
        col="#AFC0CB",
        main=paste(toString(sum(CM!=0)), "variables with missing values in dataset"))

View(as.data.frame(CM))

#Let's Formalize these ideas in one function:
dfClean <-function(df){
  # Pool Variable: If PoolQC = NA and PoolArea = 0 , assign factor NoPool
  df$PoolQC <- as.character(df$PoolQC)
  df$PoolQC[df$PoolArea %in% c(0,NA) & is.na(df$PoolQC)] <- "NoPool"
  df$PoolQC <- as.factor(df$PoolQC)
  # MiscFeature Variable: If MiscFeature = NA and MiscVal = 0, assign factor None
  df$MiscFeature <- as.character(df$MiscFeature)
  df$MiscFeature[df$MiscVal %in% c(0,NA) & is.na(df$MiscFeature)] <- "None"
  df$MiscFeature <- as.factor(df$MiscFeature)
  # Alley Variable: If Alley = NA, assign factor NoAccess
  df$Alley <- as.character(df$Alley)
  df$Alley[is.na(df$Alley)] <- "NoAccess"
  df$Alley <- as.factor(df$Alley)    
  # Fence Variable: If Fence = NA, assign factor NoFence
  df$Fence <- as.character(df$Fence)
  df$Fence[is.na(df$Fence)] <- "NoFence"
  df$Fence <- as.factor(df$Fence)
  # FireplaceQu Variable: If FireplaceQu = NA and Fireplaces = 0 , assign factor NoFirePlace
  df$FireplaceQu <- as.character(df$FireplaceQu)
  df$FireplaceQu[df$Fireplaces %in% c(0,NA) & is.na(df$FireplaceQu)] <- "NoFirePlace"
  df$FireplaceQu <- as.factor(df$FireplaceQu)   
  # GarageYrBlt Variable: If GarageYrBlt = NA and GarageArea = 0 assign factor NoGarage    
  df$GarageYrBlt <- as.character(df$GarageYrBlt)
  df$GarageYrBlt[df$GarageArea %in% c(0,NA) & is.na(df$GarageYrBlt)] <- "NoGarage"
  df$GarageYrBlt <- as.factor(df$GarageYrBlt)
  # GarageFinish Variable: If GarageFinish = NA and GarageArea = 0 assign factor NoGarage    
  df$GarageFinish <- as.character(df$GarageFinish)
  df$GarageFinish[df$GarageArea %in% c(0,NA) & is.na(df$GarageFinish)] <- "NoGarage"
  df$GarageFinish <- as.factor(df$GarageFinish)
  # GarageQual Variable: If GarageQual = NA and GarageArea = 0 assign factor NoGarage    
  df$GarageQual <- as.character(df$GarageQual)
  df$GarageQual[df$GarageArea %in% c(0,NA) & is.na(df$GarageQual)] <- "NoGarage"
  df$GarageQual <- as.factor(df$GarageQual)
  # GarageCond Variable: If GarageCond = NA and GarageArea = 0 assign factor NoGarage    
  df$GarageCond <- as.character(df$GarageCond)
  df$GarageCond[df$GarageArea %in% c(0,NA) & is.na(df$GarageCond)] <- "NoGarage"
  df$GarageCond <- as.factor(df$GarageCond)
  # GarageType Variable: If GarageType = NA and GarageArea = 0 assign factor NoGarage    
  df$GarageType <- as.character(df$GarageType)
  df$GarageType[df$GarageArea %in% c(0,NA) & is.na(df$GarageType)] <- "NoGarage"
  df$GarageType <- as.factor(df$GarageType)
  
  df$GarageArea[is.na(df$GarageArea) & df$GarageCars %in% c(0,NA)] <- 0
  df$GarageCars[is.na(df$GarageCars) & df$GarageArea %in% c(0,NA)] <- 0    
  
  
  # BsmtFullBath Variable: If BsmtFullBath = NA and TotalBsmtSF = 0 assign 0    
  df$BsmtFullBath[df$TotalBsmtSF %in% c(0,NA) & is.na(df$BsmtFullBath)] <- 0
  # BsmtHalfBath Variable: If BsmtHalfBath = NA and TotalBsmtSF = 0 assign 0   
  df$BsmtHalfBath[df$TotalBsmtSF %in% c(0,NA) & is.na(df$BsmtHalfBath)] <- 0
  
  # BsmtFinSF1 Variable: If BsmtFinSF1 = NA and TotalBsmtSF = 0 assign 0    
  df$BsmtFinSF1[df$TotalBsmtSF %in% c(0,NA) & is.na(df$BsmtFinSF1)] <- 0
  # BsmtFinSF2 Variable: If BsmtFinSF2 = NA and TotalBsmtSF = 0 assign 0   
  df$BsmtFinSF2[df$TotalBsmtSF %in% c(0,NA) & is.na(df$BsmtFinSF2)] <- 0
  # BsmtUnfSF Variable: If BsmtUnfSF = NA and TotalBsmtSF = 0 assign 0    
  df$BsmtUnfSF[df$TotalBsmtSF %in% c(0,NA) & is.na(df$BsmtUnfSF)] <- 0
  # TotalBsmtSF Variable: If TotalBsmtSF = NA and TotalBsmtSF = 0 assign 0   
  df$TotalBsmtSF[df$TotalBsmtSF %in% c(0,NA) & is.na(df$TotalBsmtSF)] <- 0
  
  # BsmtQual Variable: If BsmtQual = NA and TotalBsmtSF = 0 assign factor NoBasement    
  df$BsmtQual <- as.character(df$BsmtQual)
  df$BsmtQual[df$TotalBsmtSF %in% c(0,NA) & is.na(df$BsmtQual)] <- "NoBasement"
  df$BsmtQual <- as.factor(df$BsmtQual)
  # BsmtFinType1 Variable: If BsmtFinType1 = NA and TotalBsmtSF = 0 assign factor NoBasement    
  df$BsmtFinType1 <- as.character(df$BsmtFinType1)
  df$BsmtFinType1[df$TotalBsmtSF %in% c(0,NA) & is.na(df$BsmtFinType1)] <- "NoBasement"
  df$BsmtFinType1 <- as.factor(df$BsmtFinType1)    
  # BsmtFinType2 Variable: If BsmtFinType2 = NA and TotalBsmtSF = 0 assign factor NoBasement    
  df$BsmtFinType2 <- as.character(df$BsmtFinType2)
  df$BsmtFinType2[df$TotalBsmtSF %in% c(0,NA) & is.na(df$BsmtFinType2)] <- "NoBasement"
  df$BsmtFinType2 <- as.factor(df$BsmtFinType2)
  # BsmtExposure Variable: If BsmtExposure = NA and TotalBsmtSF = 0 assign factor NoBasement    
  df$BsmtExposure <- as.character(df$BsmtExposure)
  df$BsmtExposure[df$TotalBsmtSF %in% c(0,NA) & is.na(df$BsmtExposure)] <- "NoBasement"
  df$BsmtExposure <- as.factor(df$BsmtExposure)
  # BsmtCond Variable: If BsmtCond = NA and TotalBsmtSF = 0 assign factor NoBasement    
  df$BsmtCond <- as.character(df$BsmtCond)
  df$BsmtCond[df$TotalBsmtSF %in% c(0,NA) & is.na(df$BsmtCond)] <- "NoBasement"
  df$BsmtCond <- as.factor(df$BsmtCond)  
  return(df)    
}
df <- dfClean(df)


df$MasVnrType <- as.character(df$MasVnrType)
df$MasVnrType[is.na(df$MasVnrType)] <- "None"
df$MasVnrType <- as.factor(df$MasVnrType)  
df$MasVnrArea[is.na(df$MasVnrArea)] <- 0

df$MSZoning <- as.character(df$MSZoning)
df$MSZoning[is.na(df$MSZoning)] <- "RL"
df$MSZoning <- as.factor(df$MSZoning)

df$BsmtExposure <- as.character(df$BsmtExposure)
df$BsmtExposure[is.na(df$BsmtExposure)]<-"No"
df$BsmtExposure <- as.factor(df$BsmtExposure)

df$BsmtFinType2 <- as.character(df$BsmtFinType2)
df$BsmtFinType2[is.na(df$BsmtFinType2)]<-"ALQ"
df$BsmtFinType2 <- as.factor(df$BsmtFinType2)

df$BsmtQual <- as.character(df$BsmtQual)
df$BsmtQual[is.na(df$BsmtQual) & df$HouseStyle == "2Story"]<-"Gd"
df$BsmtQual[is.na(df$BsmtQual) & df$HouseStyle == "1.5Fin"]<-"TA"
df$BsmtQual <- as.factor(df$BsmtQual)

df$BsmtCond <- as.character(df$BsmtCond)
df$BsmtCond[is.na(df$BsmtCond)]<-"TA"
df$BsmtCond <- as.factor(df$BsmtCond)

fillMiss<- function(x){
  ux <- unique(x[!is.na(x)])
  x <- as.character(x)
  mode <- ux[which.max(tabulate(match(x[!is.na(x)], ux)))]
  x[is.na(x)] <- as.character(mode)
  x <- as.factor(x)
  return(x)
}
df[,sapply(df,function(x){!(is.numeric(x))}) ]<-as.data.frame(apply(df[,sapply(df,function(x){!(is.numeric(x))}) ],2,fillMiss))

library(ggplot2)
# Thank you Cookbook for R : http://www.cookbook-r.com/Graphs/Multiple_graphs_on_one_page_(ggplot2)/
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  numPlots = length(plots)
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  if (numPlots==1) {
    print(plots[[1]])
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

p1<-ggplot(df, aes(LotArea, LotFrontage)) + geom_point() + geom_smooth(method = "lm", se = T) 
p2<-ggplot(df, aes(log(LotArea), LotFrontage)) + geom_point() + geom_smooth(method = "lm", se = T)
p3<-ggplot(df, aes(log(LotArea), log(LotFrontage))) + geom_point() + geom_smooth(method = "lm", se = T)
p4<-ggplot(df, aes(sqrt(LotArea), LotFrontage)) + geom_point() + geom_smooth(method = "lm", se = T)
multiplot(p1, p2, p3, p4, cols=2)

cor(as.numeric(df$LotArea),as.numeric(df$LotFrontage),use="complete.obs")
cor(log(as.numeric(df$LotArea)),log(as.numeric(df$LotFrontage)),use="complete.obs")
cor(log(as.numeric(df$LotArea)),as.numeric(df$LotFrontage),use="complete.obs")
cor(sqrt(as.numeric(df$LotArea)),as.numeric(df$LotFrontage),use="complete.obs")

install.packages("outliers")
library(outliers)
chisq.out.test(df$LotArea,opposite=F)
chisq.out.test(df$LotFrontage,opposite=F)
chisq.out.test(df$LotArea,opposite=T)
chisq.out.test(df$LotFrontage,opposite=T)
grubbs.test(df$LotArea,type=11)
grubbs.test(df$LotFrontage,type=11)

plotdf = df[ !(df$LotArea %in% c(1300,215245) | df$LotFrontage %in% c(21,313) ), ]
p1<-ggplot( plotdf  , aes(LotArea, LotFrontage)) + geom_point() + geom_smooth(method = "lm", se = T) 
p2<-ggplot(plotdf, aes(log(LotArea), LotFrontage)) + geom_point() + geom_smooth(method = "lm", se = T)
p3<-ggplot(plotdf, aes(log(LotArea), log(LotFrontage))) + geom_point() + geom_smooth(method = "lm", se = T)
p4<-ggplot(plotdf, aes(sqrt(LotArea), LotFrontage)) + geom_point() + geom_smooth(method = "lm", se = T)
multiplot(p1, p2, p3, p4, cols=2)
cor(as.numeric(plotdf$LotArea),as.numeric(plotdf$LotFrontage),use="complete.obs")
cor(log(as.numeric(plotdf$LotArea)),log(as.numeric(plotdf$LotFrontage)),use="complete.obs")
cor(log(as.numeric(plotdf$LotArea)),as.numeric(plotdf$LotFrontage),use="complete.obs")
cor(sqrt(as.numeric(plotdf$LotArea)),as.numeric(plotdf$LotFrontage),use="complete.obs")








