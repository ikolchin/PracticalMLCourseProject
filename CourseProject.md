# Classifying Barbell Lifts Using Accelerometers Data



##Data Preparation
Training and test data is loaded from online storage.

```r
train.url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
train.data <- read.csv(url(train.url), na.strings=c("NA", "#DIV/0!", ""))
test.url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
test.final.data <- read.csv(url(test.url), na.strings=c("NA", "#DIV/0!", ""))
```
Training data is further subdivided into training and test subsets used respectively to train and evaluate machine learning model before making predictions for the (small) final test set.

```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
train.idx <- createDataPartition(y=train.data$classe, p=0.5, list=FALSE)
test.data <- train.data[-train.idx, ]
train.data <- train.data[train.idx, ]
dim(train.data); dim(test.data); dim(test.final.data)
```

```
## [1] 9812  160
```

```
## [1] 9810  160
```

```
## [1]  20 160
```
Irrelevant columns (e.g. those containing id, username, timestamps) are then removed. So are columns containing missing values.

```r
train.data <- train.data[, 7:160]
test.data <- test.data[, 7:160]
test.final.data <- test.final.data[, 7:160]
NA.cols <- apply(is.na(train.data), 2, any)
train.data <- train.data[!NA.cols]
test.data <- test.data[!NA.cols]
test.final.data <- test.final.data[!NA.cols]
dim(train.data); dim(test.data); dim(test.final.data)
```

```
## [1] 9812   54
```

```
## [1] 9810   54
```

```
## [1] 20 54
```

##Employing Machine Learning for Prediction
Due to the large number of features decision trees look suitable for classification task. A random forest model is fit on the training set. Fast random forest implementation in the ranger package is employed.

```r
library(ranger)

rf <- ranger(data=train.data,
             dependent.variable.name="classe",
             write.forest=TRUE)
rf
```

```
## Ranger result
## 
## Call:
##  ranger(data = train.data, dependent.variable.name = "classe",      write.forest = TRUE) 
## 
## Type:                             Classification 
## Number of trees:                  500 
## Sample size:                      9812 
## Number of independent variables:  53 
## Mtry:                             7 
## Target node size:                 1 
## Variable importance mode:         none 
## OOB prediction error:             0.49 %
```
Predictive quality of the model is then checked on the test set.

```r
predictions <- predict(rf, test.data)
confusionMatrix(predictions$predictions, test.data$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2790    2    0    0    0
##          B    0 1890   10    0    0
##          C    0    6 1701   19    0
##          D    0    0    0 1589    7
##          E    0    0    0    0 1796
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9955         
##                  95% CI : (0.994, 0.9967)
##     No Information Rate : 0.2844         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9943         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9958   0.9942   0.9882   0.9961
## Specificity            0.9997   0.9987   0.9969   0.9991   1.0000
## Pos Pred Value         0.9993   0.9947   0.9855   0.9956   1.0000
## Neg Pred Value         1.0000   0.9990   0.9988   0.9977   0.9991
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2844   0.1927   0.1734   0.1620   0.1831
## Detection Prevalence   0.2846   0.1937   0.1759   0.1627   0.1831
## Balanced Accuracy      0.9999   0.9973   0.9955   0.9937   0.9981
```
Accuracy higher than 99% is obtained. As a final step, predictions for the initial test set are generated.

```r
predictions <- predict(rf, test.final.data)
predictions$predictions
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

##Conclusion
The analysis shows that the features in the dataset allow for very high classification quality. Indeed, using only around one third of the total number of features (those not containing missing values) and fitting random forest with default parameters results in accuracy higher than 99%. As a side note, ranger implementation of random forest seems to be significantly faster than that in caret package.
