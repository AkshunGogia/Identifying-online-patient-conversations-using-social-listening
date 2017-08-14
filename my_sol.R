library(caret)
library(tm)
library(caTools)
library(xgboost)
library(RWeka)
library(dplyr)
library(tidyr)
library(data.table)
library(jsonlite)
library(purrr)
library(RecordLinkage)
library(stringr)
library(tm)


train<-read.csv("Training_Data.csv", stringsAsFactors = FALSE, na.strings = c(""," ","NA"))
test<-read.csv("Testing_Data.csv", stringsAsFactors = FALSE, na.strings = c(""," ","NA"))

setDT(train)
setDT(test)

test$Patient_Tag<-NA

tr <- train[,.(Patient_Tag,TRANS_CONV_TEXT,Title)]
ts <-test[,.(Patient_Tag,TRANS_CONV_TEXT,Title)]

tdata <- rbindlist(list(tr,ts))

tdata[,Trans_count := str_count(TRANS_CONV_TEXT,pattern = "\\w+")]
tdata[,Trans_len := str_count(TRANS_CONV_TEXT)]

tdata[, Title_count:= str_count(Title,pattern = "\\w+")]
tdata[,Title_len := str_count(Title)]

tdata[,lev_Text_Title := levenshteinDist(TRANS_CONV_TEXT,Title)]


text_corpus <- Corpus(VectorSource(tdata$TRANS_CONV_TEXT))
inspect(text_corpus[1])

text_corpus <- tm_map(text_corpus, tolower)
text_corpus <- tm_map(text_corpus, removePunctuation)
text_corpus <- tm_map(text_corpus, removeNumbers)
text_corpus <- tm_map(text_corpus, stripWhitespace)
text_corpus <- tm_map(text_corpus, removeWords, c(stopwords('english')))
text_corpus <- tm_map(text_corpus, stemDocument,language = "english")



#Trigram function
Trigram_Tokenizer <- function(x){
  NGramTokenizer(x, Weka_control(min=3, max=3))
}

#create a matrix
docterm_corpus <- DocumentTermMatrix(text_corpus, control = list(tokenize = Trigram_Tokenizer))
dim(docterm_corpus)


new_docterm_corpus <- removeSparseTerms(docterm_corpus,sparse = 0.99)
dim(new_docterm_corpus)


colS <- colSums(as.matrix(new_docterm_corpus))
length(colS)

doc_features <- data.table(name = attributes(colS)$names, count = colS)

doc_features[order(-count)][1:10] #top 10 most frequent words
doc_features[order(count)][1:10] #least 10 freuqnet
 
library(wordcloud)
#this wordcloud is exported
wordcloud(names(colS), colS, min.freq = 5000, scale = c(6,.1), colors = brewer.pal(6, 'Dark2'))


processed_data <- as.data.table(as.matrix(new_docterm_corpus))
data_one <- cbind(data.table(Patient_Tag = tdata$Patient_Tag),processed_data)

#split the data set into train and test
train_one<-filter(data_one, Patient_Tag !="NA")
test_one <- data_one[1158:nrow(tdata),]

train$Patient_Tag<-as.factor(train_one$Patient_Tag)


#returns 70% indexes from train data
sp <- sample.split(Y = train_one$Patient_Tag ,SplitRatio = 0.7)


setDT(train_one)
setDT(test_one)

#create data for xgboost
xg_val <- train_one[sp]
target <- train_one$Patient_Tag
xg_val_target <- target[sp]


d_train <- xgb.DMatrix(data = as.matrix(train_one[,-c("Patient_Tag")]),label = target)
d_val <- xgb.DMatrix(data = as.matrix(xg_val[,-c("Patient_Tag")]), label = xg_val_target)
d_test <- xgb.DMatrix(data = as.matrix(test_one[,-c("Patient_Tag")]))


param <- list(booster="gbtree",
              objective="reg:logistic",
              gamma=5,
              eval_metric="auc",
              eta = 0.02)

set.seed(2017)

watch <- list(val=d_val, train=d_train)

xgb <- xgb.train(data = d_train,
                  params = param,
                  watchlist=watch,
                  nrounds = 400,
                  print_every_n = 10)


pred_test<-matrix(predict(xgb, d_test))
pred_train<-matrix(predict(xgb, d_train))

fitted.results<-pred_test
fitted.results2<-pred_train

fitted.results <- ifelse(fitted.results >= 0.5,1,0)
fitted.results2 <- ifelse(fitted.results2 >= 0.5,1,0)

confusionMatrix(fitted.results2,train_one$Patient_Tag)

#view variable importance plot
mat <- xgb.importance (feature_names = colnames(train_one),model = xgb)
xgb.plot.importance (importance_matrix = mat[1:20]) 

test$Patient_Tag<-NULL
final<-cbind(test, fitted.results)
setnames(final,"V1", "Patient_Tag")


library(ROCR)
ROCRpred <- prediction(pred_train, train_one$Patient_Tag)
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2,1.7))

auc <- performance(ROCRpred, measure = "auc")
auc <- auc@y.values[[1]]
auc

