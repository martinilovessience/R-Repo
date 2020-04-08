#### Bayer AG Data Science Interview ####
### Sentiment analysis


#install.packages("ggplot2")
#install.packages("magrittr")
#install.packages("dplyr")
#install.packages(c('e1071', 'rpart'))
#install.packages("tibble")
#install.packages("dplyr")
#install.packages("tm")
#install.packages("wordcloud")
#install.packages("RColorBrewer")

library(RTextTools)
library(dplyr)
library(e1071)
library(rpart)
library(tm)
library(openxlsx)
library(ggplot2)
library(dplyr)
library(wordcloud)
library(RColorBrewer)


setwd("/Users/martinkurek/Documents/R Repo/bayer_case")


## Part 1: Data integration and transformation

# Read in and load data
sm_data <- read.xlsx("data/sentences_with_sentiment.xlsx")
hist_data <- as.data.frame(sm_data[, 3:5])
tdata <- as.data.frame(sm_data[, 2])
sent <- assignSentiment(hist_data)
full_data <- cbind(tdata, sent)

# Declare functions

# List of words function can be used for selection of features but was on purpose not applied due to lack of words.
#  listOfWords<- {c(
#    "pmlast"
#  )}

# Transform binary scema into nominal
assignSentiment <- function(x) {
  i = 1
  y <- data.frame()
  
  for (i in i:nrow(hist_data)) {
    if (hist_data[i, 1] == 1) {
      y[i, 1] = "Positive"
    } else if (hist_data[i, 2] == 1) {
      y[i, 1] = "Negative"
    } else {
      y[i, 1] = "Neutral"
    }
  }
  
  return(y)
}


# Generate Document Term Matrix with specified text preprocessing criteria

generateDTM <- function(z) {
  y <- VCorpus(VectorSource(z[, 1]))
  
  y <- tm_map(y, content_transformer(tolower))
  
  y <- tm_map(y, removeNumbers)
  
  y <- tm_map(y, removePunctuation)
  
  y <- tm_map(y, removeWords, stopwords("english"))
  
  #y <- tm_map(y, removeWords, listOfWords)
  
  y <- tm_map(y, stripWhitespace)
  
  y <- DocumentTermMatrix(x = y)
  
  return(y)
}

# Generate Document Term Matrix with same criteria

generateCorpus <- function(z) {
  y <- VCorpus(VectorSource(z[, 1]))
  
  y <- tm_map(y, content_transformer(tolower))
  
  y <- tm_map(y, removeNumbers)
  
  y <- tm_map(y, removePunctuation)
  
  y <- tm_map(y, removeWords, stopwords("english"))
  
  #y <- tm_map(y, removeWords, listOfWords)
  
  y <- tm_map(y, stripWhitespace)
  
  return(y)
}


## Part 2: Data exploration
Sentiment <- c("Positive", "Negative", "Neutral")
Count <-
  c(sum(hist_data$Positive),
    sum(hist_data$Negative),
    sum(hist_data$Neutral))
total_count = sum(hist_data$Positive) + sum(hist_data$Negative) + sum(hist_data$Neutral)
Percent <- c(paste(round((
  sum(hist_data$Positive) / total_count
) * 100), "%"),
paste(round((
  sum(hist_data$Negative) / total_count
) * 100), "%"),
paste(round((
  sum(hist_data$Neutral) / total_count
) * 100), "%"))
hist_sum <- data.frame(Sentiment, Count, Percent)

summary(sm_data)

#Plot distribution with ggplot

g <- ggplot(hist_sum, aes(Sentiment, Count))
g + geom_col(fill = "blue") + geom_label(label = Percent)



### Part 3: Model training

dtm.train <- generateDTM(tdata)

corupus.train <- generateCorpus(tdata)

##Split into Train & Testset

index <- 1:nrow(sent)

# Set Parameter for sampling (75% Training, 25% Testing) by defining and applying index

testindex <- sample(index, trunc(length(index)) / 1.333)

df.train <- sent[testindex,]
df.test <- sent[-testindex,]

trainset <- dtm.train[testindex,]
testset <- dtm.train[-testindex,]

corpus.ts <- corupus.train[testindex]
corpus.tss <- corupus.train[-testindex]

## Part 4: Feature selection by chosing frequency interval

dim(trainset)

# finding terms with lowest 2 and highest 6 frequency
ft <- findFreqTerms(trainset, 1, 6)

#Wordcould
freq_tf<- data.frame(sort(x = colSums(x = as.matrix(trainset)), decreasing = TRUE))
wcloud<- wordcloud(words = rownames(x = freq_tf), freq = freq_tf[,1], max.words = 100, scale = c(1,.6), colors = brewer.pal(3, "GnBu"))


length((ft))

trainset <-
  DocumentTermMatrix(corpus.ts, control = list(dictionary = ft))
testset <-
  DocumentTermMatrix(corpus.tss, control = list(dictionary = ft))


##Train model with SVM 

container <-
  create_container(trainset,
                   df.train,
                   trainSize = 1:nrow(trainset),
                   virgin = FALSE)

##Linear Kernel because values are either 0 or 1 and evaluation of applying both showed linear fits better
pred.model <-
  train_model(container = container,
              algorithm = "SVM",
              kernel = "linear")


predicted <- predict(pred.model, newdata = testset)


## Part 5: Evaluation 
table <- table("Predictions" = predicted, "Actual" = df.test)

#Print Confusion Matrix
print(table)

#Precision
nB_precision_ng <- round(table[1, 1] / sum(table[, 1]), 2)
nB_precision_nt <- round(table[2, 2] / sum(table[, 2]), 2)
nB_precision_p <- round(table[3, 3] / sum(table[, 3]), 2)

#Recall
nB_recall_ng <- round(table[1, 1] / sum(table[1, ]), 2)
nB_recall_nt <- round(table[2, 2] / sum(table[2, ]), 2)
nB_recall_p <- round(table[3, 3] / sum(table[3, ]), 2)


#Print Precision and Recall- Measures

print(c("Precision class 'Negative':", nB_precision_ng))
print(c("Recall class 'Negative':", nB_recall_ng))
print(c("Precision class 'Positive':", nB_precision_p))
print(c("Recall class 'Positive':", nB_recall_p))
print(c("Precision class 'Neutral':", nB_precision_nt))
print(c("Recall class 'Neutral':", nB_recall_nt))

#Print overall accuracy
print(paste("Accuracy:", (round(table[1, 1]) + round(table[2, 2]) + round(table[3,3]))/ (sum(table[1, ]) + sum(table[2, ]) +  sum(table[3, ]))))
