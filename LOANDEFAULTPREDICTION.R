library(RODBC)
library(odbc)
library(corrplot)
library(dplyr)
library(tidyverse)
library(corrplot)
library(modeldata)
library(Metrics)
library(ggeffects)
library(leaps)  # helps with function StepAIC
con <- dbConnect(odbc::odbc(),
                 Driver = "SQL Server",
                 Server = "DESKTOP-2F3Q272\\SQLEXPRESS",
                 Database = "LoanDefaultPrediction",
                 Port = 1433)

dbListTables(con)
dbListFields(con, 'Data_train')

testdata <- tbl(con, 'Data_train')
testdata <-collect(testdata)
View(testdata)

#This is to convert Variables to numerics

testdata$Due_Fee<- as.integer(testdata$Due_Fee)
testdata$Unpaid_Amount<- as.integer(testdata$Unpaid_Amount)
testdata$Gross_Collection<-as.integer(testdata$Gross_Collection)
testdata$Inquiries<- as.integer(testdata$Inquiries)
testdata$Usage_Rate<-as.integer(testdata$Usage_Rate)
testdata$Deprecatory_Records<- as.integer(testdata$Deprecatory_Records)
testdata$Debt_to_Income<- as.integer(testdata$Debt_to_Income)
testdata$Already_Defaulted<- as.integer(testdata$Already_Defaulted)
testdata$Unpaid_2_years<- as.integer(testdata$Unpaid_2_years)


colnames(testdata)




#Factoring our predictors
testdata$Loan_No_Loan<- as.factor(testdata$Loan_No_Loan)

testdata$Validation<-factor(testdata$Validation)
levels(testdata$Validation)


testdata$Claim_Type<- factor(testdata$Claim_Type)
levels(testdata$Claim_Type)


#Dropping features from the train data set
drop<- c("State", "Postal_Code","Home_Status", "Designation","Experience", "GGGrade","Reason","Duration", "File_Status", "ID")
testdata= testdata[, !(names(testdata) %in% drop)]

str(testdata)

#Removing null values from the train data
df_clean<- na.omit(testdata)
str(df_clean)




#Training and testing my model using test, train split

indexset<- sample(2, nrow(df_clean), replace = T, prob = c(0.8, 0.2))
set.seed(1234) #This is to ensure reproducibility

train <- df_clean[indexset==1,]
View(train)

test<-df_clean[indexset==2,]
View(test)

lapply(df_clean, class)




#USING RANDOMFOREST, THE RESULTS FROM THE LOGISTIC REGRESSION WERE UNSATISFACTORY

library(ROSE)
library(caret)
library(e1071)
library(randomForest)

summary(testdata$Loan_No_Loan)
summary(testdata)

barplot(prop.table(table(testdata$Loan_No_Loan)), col= rainbow(2), ylim=c(0,1), main= 'Class Distribution')

#Based on the plot, it is clearly evident that 80% of the data in one class and the remaining 30% in another class



# We notice a class imbalance in our data train data, 81% for 0 while 19% for 1

prop.table(table(train$Loan_No_Loan))
(train)

#Training our model with Randomforest classifier

rftrain<- randomForest(train$Loan_No_Loan~., data=train)
summary(rftrain)

#Cross validating based on test data

confusionMatrix(predict(rftrain, test), test$Loan_No_Loan, positive= '1')

#install.packages("caret", dependencies=c("Depends", "Imports"))

#We notice that the sensitivity is just 27% which clearly shows that one of the classes is being dominated by the other

#We will over sample

df_clean[is.na(df_clean)] <- 0
train[is.na(train)]<-0
over<-ovun.sample(Loan_No_Loan~.,  data=train, method= "over", N=56601)$data
table(over$Loan_No_Loan)


summary(over)


#Building the Randomforest model
rfover <- randomForest(Loan_No_Loan~., data = over)

confusionMatrix(predict(rfover, test), test$Loan_No_Loan, positive = '1')


#Under sampling: We notice that undersampling gives a better accuracy and sensitivity than over sampling
under <- ovun.sample(Loan_No_Loan~., data=train, method = "under", N = 56601)$data
table(under$Loan_No_Loan)

rfunder <- randomForest(Loan_No_Loan~., data=under)
confusionMatrix(predict(rfunder, test), test$Loan_No_Loan, positive = '1')


#Both (oversampling and under sampling): We notice that over and under sampling gives us a lower accuracy but a much higher sensitivity
#Both under and over sampling will be used because they give it gives us a much preferable model


both <- ovun.sample(Loan_No_Loan~., data=train, method = "both",
                    p = 0.5,
                    seed = 222,
                    N = 56601)$data
table(both$Loan_No_Loan)

rfboth <-randomForest(Loan_No_Loan~., data=both)
confusionMatrix(predict(rfboth, test), test$Loan_No_Loan, positive = '1')


#Using the model to make predictions on the test dataset
dbListFields(con, 'Data_Test')

traindata <- tbl(con, 'Data_Test')
traindata <-collect(traindata)
View(traindata)


#Changing the data type of numeric features to integers
traindata$Due_Fee<- as.integer(traindata$Due_Fee)
traindata$Unpaid_Amount<- as.integer(traindata$Unpaid_Amount)
traindata$Gross_Collection<-as.integer(traindata$Gross_Collection)
traindata$Inquiries<- as.integer(traindata$Inquiries)
traindata$Usage_Rate<-as.integer(traindata$Usage_Rate)
traindata$Deprecatory_Records<- as.integer(traindata$Deprecatory_Records)
traindata$Debt_to_Income<- as.integer(traindata$Debt_to_Income)
traindata$Already_Defaulted<- as.integer(traindata$Already_Defaulted)
traindata$Unpaid_2_years<- as.integer(traindata$Unpaid_2_years)

str(traindata)

colnames(traindata)


#Giving weight to our numeric variables using factoring
#traindata$Loan_No_Loan<- as.factor(traindata$Loan_No_Loan)

traindata$Validation<-factor(traindata$Validation)
levels(traindata$Validation)



traindata$Claim_Type<- factor(traindata$Claim_Type)
levels(traindata$Claim_Type)


#Dropping features we do not need for our model building
drop<- c("State", "Postal_Code","Home_Status", "Designation","Experience", "GGGrade","Reason","Duration", "File_Status")
traindata2= traindata[, !(names(traindata) %in% drop)]

clean_test<- na.omit(traindata2)


drop2<-("ID")
traindata2= clean_test[, !(names(clean_test) %in% drop2)]



#Taking out null values from the dataset
dff_clean<- na.omit(traindata2)
str(dff_clean)


#Testing model on whole train split
confusionMatrix(predict(rfboth, testdata), testdata$Loan_No_Loan, positive = '1')



#Prediction
finalpredict<- rfboth %>% predict(dff_clean)



#The error rate of trees could not be improved after about 100 trees

plot(rfboth)


summary(rfboth)


hist(treesize(rfboth), main = "No. of Nodes for the trees", col="red")

#Variable Importance
varImpPlot(rfboth)
importance(rfboth)
varUsed(rfboth)

#Partial Dependence
partialPlot(rfboth, dff_clean, Lend_Amount, "1")

#Extract single tree
getTree(rfboth,1, labelVar = TRUE)


#Setting prediction into a dataframe
Predictedd<-data.frame(ID= clean_test$ID, Predicted=finalpredict)

View(predictedvalues)


