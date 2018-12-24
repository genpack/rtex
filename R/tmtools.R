# Header
# Filename:     tmtools.R
# Version History:
# Version   Date                Action
# ---------------------------------------
# 2.0.0     19 November 2015       function text.vect.to.term.document.matrix() Applies a given conversion dictionary before creating the corpus

# This url is helpful to generate a workspace containing all english words
# http://www.manythings.org/vocabulary/lists/l/words.php?f=3esl.08

library(tm)
library(stringr)

url.parts <- function(x) {
  ## returns parts of a URL:
  m <- regexec("^(([^:]+)://)?([^:/]+)(:([0-9]+))?(/.*)", x)
  parts <- do.call(rbind,
                   lapply(regmatches(x, m), `[`, c(3L, 4L, 6L, 7L)))
  colnames(parts) <- c("protocol","host","port","path")
  parts
}

remove.special.charachters <- function(str.vect){
  return(gsub("[[:punct:]]", " ", str.vect))
}

text.vect.to.term.document.matrix = function(tv, extra_stopwords = c(), unique = TRUE){
  #Converts a vector of texts into a frequency matrix
  #Step 1: remove special characters from the given text vector
  tv = remove.special.charachters(tv)
  if (unique){tv = unique(tv)}
  #Step 2: construct a corpus out of the given text vector
  crp = Corpus(VectorSource(tv))
  #Step 2: make necessary modifications
  stoplist=c(stopwords('english'), letters, extra_stopwords)
  tdm = TermDocumentMatrix(crp,control = list(removePunctuation = TRUE, stopwords = stoplist,removeNumbers = TRUE, tolower = TRUE,stemming = TRUE))
  return(tdm)
}

text.vect.to.document.term.matrix = function(tv, extra_stopwords = c(), unique = TRUE, dictionary = NA){
  #Converts a vector of texts into a frequency matrix
  #Step 1: remove special characters from the given text vector
  tv = remove.special.charachters(tv)
  if (unique){tv = unique(tv)}
  # Step 2: construct a corpus out of the given text vector
  crp = Corpus(VectorSource(tv))

  # Step 3: Clean the text from punctuations and numbers and convert all to lower case
  crp <- tm_map(crp, removePunctuation) # remove punctuation
  crp <- tm_map(crp, removeNumbers) # remove numbers
  crp <- tm_map(crp, content_transformer(tolower)) #lower case

  # Step 3: Make dictionary conversions
  if (inherits(dictionary,'data.frame')){
    for (j in seq(crp)){
      for (i in 1:(dim(dic)[1])){
        crp[[j]]$content <- gsub(paste0('\\<', dic[i,1] , '\\>'), dic[i,2], crp[[j]]$content)
      }
    }
  }

  #Step 4: remove standard English stopwords, extra stopwords,
  stoplist=c(stopwords('english'), letters, extra_stopwords)
  crp <- tm_map(crp, removeWords, stoplist)
  crp <- tm_map(crp, PlainTextDocument) # make sure it's read as plain text

  #crp <- tm_map(crp, stemDocument)

  # Step 5: Do standard English stemming and convert to Term Document Matrix

  dtm <- DocumentTermMatrix(crp, control = list(minWordLength = 1))

  return(dtm)
}

text.vect.to.frequency.matrix = function(tv, extra_stopwords = c(), unique = TRUE, dictionary = NA){
  #Converts a vector of texts into a frequency matrix
  tdm = text.vect.to.term.document.matrix(tv, extra_stopwords = extra_stopwords, unique = unique, dictionary = dictionary)
  return(t(as.matrix(tdm)))
}

sensitivity <- function(predicted.class, actual.class) {
  return(mean(predicted.class[actual.class]))
}

naive.bayes.test <- function(data){

  #step 1: extract positive and negative documents
  pn.index   = which(data$sentiment == "positive" | data$sentiment == "negative")
  texts      = data$text[pn.index]
  sentiments = data$sentiment[pn.index]
  # Step 2: Make a frequency matrix out of the text vector
  A = text.vect.to.frequency.matrix(texts)
  # Step 3: Split train and validate texts randomly
  n.texts = dim(A)[1]
  train.index = sample(n.texts, floor(n.texts/2))
  A.train = A[train.index,]
  C.train = sentiments[train.index]
  A.validate = A[- train.index,]
  C.validate = sentiments[- train.index]
  # Step 4: Find the bias for the training model
  p.positive = mean(C.train == "positive")
  p.negative = 1.0 - p.positive
  bias = log(p.positive/p.negative)
  # Step 5: Find sentiment weightings for the words:

  #   Step 5.0: change train frequencies according to the rule of success:
  A.train = A.train + 1
  #   Step 5.1: Find the word frequencies in positive texts
  word.freqs.in.positive.texts = colSums(A.train[C.train=="positive",])
  total.number.of.words.in.positive.texts =sum(word.freqs.in.positive.texts)
  #   Step 5.2: Find the word probabilities in positive texts
  word.probs.positive = word.freqs.in.positive.texts/total.number.of.words.in.positive.texts

  #   Step 5.3: Find the word frequencies in negative texts
  word.freqs.in.negative.texts = colSums(A.train[C.train=="negative",])
  total.number.of.words.in.negative.texts =sum(word.freqs.in.negative.texts)
  #   Step 5.2: Find the word probabilities in negative texts
  word.probs.negative = word.freqs.in.negative.texts/total.number.of.words.in.negative.texts
  #   Step 5.3: Find the word weights
  word.weights = log(word.probs.positive/word.probs.negative)

  # Step 6: Compute sentiments of validation data

  scores=c()
  n.texts = length(C.validate)
  for(i in 1:n.texts){
    word.index = which(A.validate[i, ] > 0)
    scores[i] = sum(word.weights[word.index]) + bias
  }

  mid.score  = median(scores)
  sentiments.computed = (scores > mid.score)
  sentiments.observed = (C.validate == "positive")

  # Step 7: Statistical Analysis
  num.suc.preds = n.texts - sum(xor(sentiments.computed, sentiments.observed))
  suc.rate = num.suc.preds/n.texts
  sd.proportion = sqrt(suc.rate*(1.0 - suc.rate)/n.texts)
  conf.int = range(suc.rate - 1.96*sd.proportion, suc.rate + 1.96*sd.proportion)
  # Step 8: Compute Sensitivity and Specificity
  sensitive = mean(sentiments.computed[sentiments.observed])
  specific  = mean(!sentiments.computed[!sentiments.observed])
  # Step 9: Issue the output
  output = list(number.of.successful.predictions = num.suc.preds, out.of = n.texts, success.rate = suc.rate, confidence.interval = conf.int, sensitivity=sensitive, specificity = specific)
  return(output)
}


# binary.bayesian.classification.model <- function(training.text.vector, training.sentiments)
#   # This function generates a binary bayesian classification model from thegiven taining data
#   # Input 1: training.text.vector - A vector of texts (characters) containing the texts used for training the model
#   # Input 2: training.sentiments - A vector of booleans containing TRUE and FALSE as a binary outcome of its equivalent text
#   # Input 1 & Input 2 must have the same length. The following piece of code checks this:
#   if (!vector.dimension.equal(training.text.vector, training.sentiments)){
#     print("binary.bayesian.classification.model Error: Given vectors must have the same length")
#     return(NA)
#   }
#   # Output: A list of variables:
#   # Output$words: a vector of strings containing all the words in the given documents
#   # Output$weightings: a vector of conditional probabilities P(X = x_i|C = 1) for each word
#   # output$bias: bias of the model
#
#   p.C.1 = mean(training.sentiments)
#   p.c.0 = 1 - p.c.1
#
#   #Convert the training text vector into a frequency matrix:
#   A = str.vect.to.frequency.matrix(training.text.vector)
#   n.words = ncols(A)
#   # this function is not complete yet. Complete it in the future
#
# predict.sentiment <- function(bin.bayes.model, text)
#   #Input 1: bin.bayes.model - the output of binary.bayesian.classification.model
#   #Input 2: text - document that you want to predict its sentiment
#   # this function is not complete yet. Complete it in the future
#

binary.metric <- function(x, y) {
  return(mean(xor(x, y)))
}

text.distance <- function(text, train.texts) {
  # text is a row vector from A.validate, train.texts is A.train

  # return a vector of distances between text and all vectors in
  # train.texts measured using the function binary.metric
  td=c()
  n.texts=dim(train.texts)[1]
  for (i in 1:n.texts){
    td[i] = binary.metric(text, train.texts[i,])
  }
  return(td)
}

kNN.classify <- function(text, train.texts, train.classes, k = 10) {
  # identify the k closest texts
  d = text.distance(text, train.texts)
  closest.k.text.positions = order(d)[1:k]

  # select classes of closest texts
  close.classes = train.classes[closest.k.text.positions]

  # return the most frequently appearing class (the mode)
  t = table(close.classes)
  return(names(which.max(t)))
}

kNN.predict.classes <- function(validate.texts, train.texts, train.classes, k = 10){

  n.texts=dim(validate.texts)[1]
  classes=c()
  for(i in 1:n.texts){
    print(i)
    classes[i] = kNN.classify(validate.texts[i,], train.texts, train.classes, k = k)
  }
  return(classes)
}

test.kNN.classifier <- function(data, k = 10){

  #step 1: extract positive and negative documents
  pn.index   = which(data$sentiment == "positive" | data$sentiment == "negative")
  texts      = data$text[pn.index]
  sentiments = data$sentiment[pn.index]
  # Step 2: Make a frequency matrix out of the text vector
  A = text.vect.to.frequency.matrix(texts)
  # Step 3: Split train and validate texts randomly
  n.texts = dim(A)[1]
  train.index = sample(n.texts, floor(n.texts/2))
  A.train = A[train.index,]
  C.train = sentiments[train.index]
  A.validate = A[- train.index,]
  C.validate = sentiments[- train.index]
  # Step 4: Classify the training data
  predicted.classes = kNN.predict.classes(A.validate, A.train, C.train, k = k)

  sentiments.computed = (predicted.classes == "positive")
  sentiments.observed = (C.validate == "positive")

  # Step 5: Statistical Analysis
  n.texts = length(C.validate)
  num.suc.preds = n.texts - sum(xor(sentiments.computed, sentiments.observed))
  suc.rate = num.suc.preds/n.texts
  sd.proportion = sqrt(suc.rate*(1.0 - suc.rate)/n.texts)
  conf.int = range(suc.rate - 1.96*sd.proportion, suc.rate + 1.96*sd.proportion)
  # Step 8: Compute Sensitivity and Specificity
  sensitive = mean(sentiments.computed[sentiments.observed])
  specific  = mean(!sentiments.computed[!sentiments.observed])
  # Step 9: Issue the output

  output = list(number.of.successful.predictions = num.suc.preds, out.of = n.texts, success.rate = suc.rate, confidence.interval = conf.int, sensitivity=sensitive, specificity = specific)
  return(output)
}


SimpleWordle <- function(words, freq, min.freq=10) {
  keep <- (freq >= min.freq)
  words <- words[keep]
  freq <- freq[keep]

  ord <- order(freq, decreasing=TRUE)
  freq <- freq[ord]
  words <- words[ord]

  plot.new()
  op <- par(mar=rep(0,4))
  plot.window(c(-1,1),c(-1,1), asp=1)

  smin <- 0.5
  smax <- 4
  sizes <- smin + (smax-smin) *(freq-min(freq))/(max(freq)-min(freq))

  thetaStep <- 0.1
  rStep <- 0.05*thetaStep/(2*pi)
  boxes <- list()

  box <- function(r, theta, word, size) {
    wid <- strwidth(word, cex=size)
    ht <- strheight(word, cex=size)
    x <- r*cos(theta)
    y <- r*sin(theta)
    return(c(x-wid/2, x+wid/2, y-ht/2, y+ht/2))
  }

  is.overlapped <- function(box, boxes) {
    if(length(boxes)==0) return(FALSE)
    for(i in 1:length(boxes)) {
      boxi <- boxes[[i]]
      if(boxi[1]>box[2]  || boxi[2]<box[1] || boxi[3]>box[4] || boxi[4] < box[3]) next
      else return(TRUE)
    }
    return(FALSE)
  }
  r <- 0
  theta <- 0
  for(i in 1:length(words)) {
    repeat {
      b<-box(r, theta, words[i], sizes[i])
      if(!is.overlapped(b, boxes)) break
      theta <- theta + thetaStep
      r <- r + rStep
    }
    text(r*cos(theta),r*sin(theta), words[i], adj=c(0.5,0.5), cex=sizes[i])
    boxes <- c(list(b), boxes)
  }
  par(op)
  invisible()
}

