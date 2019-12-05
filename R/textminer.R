# Header
# Filename:     textminer.R
# Version History:
# Version   Date                Action
# ---------------------------------------
# 3.1.0     10 March 2016       Modified to accommodate gener version 1.2
# 3.1.1     26 September 2016   Global variable 'metrics' and 'weightings' exported and renamed to 'valid.metrics' and 'valid.weightings'
# 3.2.0     05 October 2016     Documentation added for some methods. Some slot names changed.
# 3.2.1     07 October 2016     Plot methods renamed and combined.
# 3.2.2     07 October 2016     Method plot.mds.2d() supports rCharts scatter plots calling niraPlotter function from niravis.
# 3.2.3     07 October 2016     Method plot.wordCloud() supports package wordcloud2.
# 3.2.4     07 October 2016     Methods subset() and subsets() renamed to subsetObject() and subsetObjects().
# 3.2.5     07 October 2016     Methods cluster() and clusters() renamed to clusterObject() and clusterObjects().
# 3.3.0     22 February 2018    Fundamental structural change in the TEXT.MINER class:
# Package text2vec can now be used for the creation of dtm matrix
# All properties moved to list data, text vector is now a column of table dataset. This table contains many other data regarding the text (case)
# Method names changed: get.docterm() renamed to get.dtm()
# 3.3.1     01 March 2018       Method get.prc() for Principal Component Analysis added.
# 3.3.2     01 March 2018       Methods plot.prc.2d() and plot.prc.3d() added.
# 3.3.3     01 March 2018       Metric 'jaccard' added.
# 3.3.4     13 March 2018       Some bugs fixed.
# 3.4.0     14 March 2018       Topic Modeling added: Methods: get.doctopic(), get.topicword(), get.themes() and plot.topics() added.


valid.metrics    = c("euclidean", "maximum", "manhattan", "canberra", "binary" , "minkowski", "spherical", "jaccard")

valid.weightings = c("tfidf", "freq")

plural.col = function(color){
  #   "BrBG"     "PiYG"     "PRGn"     "PuOr"     "RdBu"
  #   "RdGy"     "RdYlBu"   "RdYlGn"   "Spectral" "Accent"
  #   "Dark2"    "Paired"   "Pastel1"  "Pastel2"  "Set1"
  #   "Set2"     "Set3"     "Blues"    "BuGn"     "BuPu"
  #   "GnBu"     "Greens"   "Greys"    "Oranges"  "OrRd"
  #   "PuBu"     "PuBuGn"   "PuRd"     "Purples"  "RdPu"
  #   "Reds"     "YlGn"     "YlGnBu"   "YlOrBr"   "YlOrRd"

  if      (color == 'black') {return('Greys')}
  else if (color == 'blue') {return('Blues')}
  else if (color == 'red') {return('Reds')}
  else if (color == 'green') {return('Greens')}
  else if (color == 'orange') {return('Oranges')}
  else if (color == 'purple') {return('Purples')}
  else {return(color)}
}

# Private method: Do not export
arg.verify = function(weighting, metric){
  assert(metric %in% valid.metrics, "Error from get.dist(): Argument metric is unknown")
  assert(weighting %in% valid.weightings, "Error from get.dist(): Argument weighting is unknown")
}

#' A list of default settings for a TEXT.MINER object:
#' @field remove_punctuation a single logical: Should punctuations be removed from all text documents? (default is TRUE)
#' @field remove_numbers a single logical: Should numbers be removed from all text documents? (default is TRUE)
#' @field tolower a single logical: should all letters be converted to lower case? (default is TRUE)
#' @field stemming a single logical: should all words be reduced to their stem? (default is FALSE)
#' @field remove_special_characters logical: should all special characters be removed? (default is TRUE)
#' @field plain_text a single logical: should all the documents be treated as plain text? (default is TRUE)
#' @field unique a single logical: should duplicated documents be removed? (default is TRUE)
#' @field weighting a single character: specifies the default weighting. Must be within \code{c('freq', 'tfidf')}. (default is \code{'tfidf'})
#' @field metric a single character: specifies the default metric for computing distances between the documents.
#' Must be within \code{c("euclidean", "maximum", "manhattan", "canberra", "binary" , "minkowski", "spherical")}.
#' (default is \code{'spherical'})
#' @field wc_max_words a single integer: specifies the maximum number of words shown in the word cloud.
#' @field wc_rot_per a single numeric: must be between 0 and 1. Specifies the percentage of words shown as rotated in the word cloud.
#' @field wc_color a single character: specifies the color of the words shown in the word cloud.
#' @field wc_gradient a single character: which weighting should be reflected by the color gradient in the word cloud.
#' Must be within \code{c('freq', 'tfidf')}
#' @field wc_color a single character: specifies the color of the points in the point 2d and 3d plots.
#' @field num_clust a single integer: specifies the default number of clusters. (default is 3)
#' @field sparsity a single numeric: must be between 0 and 1 and specifies the sparsity.
#' For example, if sparcity is 0.98, all words appearing in less than 2\% of the documents will be removed. (default is 0.99)
#'
#' @export
genDefaultSettings = function(remove_punctuation = TRUE, remove_numbers = TRUE,
                              tolower = TRUE, metric = 'spherical', stemming = TRUE,
                              remove_special_characters = TRUE, plain_text = TRUE,
                              unique = TRUE,
                              weighting = 'freq', wc_max_words = 50,
                              wc_rot_per = 0.4, stop_words = c(letters, LETTERS, tm::stopwords('english')),
                              wc_color = 'blue', num_clust = 3, wc_gradient = 'weight', dictionary = data.frame(),
                              plot_color = 'blue', sparsity = 0.999){
  list(remove_punctuation = remove_punctuation, remove_numbers = remove_numbers,
       tolower = tolower, metric = metric, stemming = stemming,
       remove_special_characters = remove_special_characters, plain_text = plain_text,
       unique = unique, weighting = weighting, wc_max_words = wc_max_words, stop_words = stop_words,
       wc_rot_per = wc_rot_per, wc_color = wc_color, num_clust = num_clust, wc_gradient = wc_gradient,
       plot_color = plot_color, sparsity = sparsity)
}

#' Reference Class TEXT.MINER is a combination of properties and methods for running various text mining
#' algorithms
#'
#' @field text vector of character containing raw text documents which are contents of argument \code{text_vect} passed to the class constructor.
#' @field n.text a single integer indicating the count of text documents.
#' @field stop.words vector of character specifying words to be removed from the text corpus.
#' @field dictionary data.frame of two columns containing words to be replaced with their synonyms.
#' Words in the first column are replaced by the words in the second.
#' @field data$words vector of character containing the words in all the documnets.
#' @field time vector of POSIXlt containing the time in which the text document is issued. Need to be given to the class constructor as an argument.
#' @field settings list of various parameters containing the settings of the text miner object:
#'
#' @field data$DTM a matrix of numerics representing the document term matrix of the text corpus.
#' Better to use method get.dtm() to get the matrix.
#' @field data$W.tfidf matrix of numerics containing the tf-idf weights of the document-term matrix.
#' Better to use method get.tfidf() to get the matrix.
#' @field data$W.bin matrix of numerics containing the binary weights of the term-document matrix.
#' @field D.bin matrix \code{Nd x Nd} of numerics, where \code{Nd} is the number of documents.
#' Contains the distances of all pairs of documents based on \emph{binary} metric.
#' @field data$D.freq.euc matrix same size as \code{D.bin matrix}.
#' Contains the distances of all pairs of documents based on euclidean metric using raw frequencies as word weights.
#' @field data$D.freq.max matrix same size as \code{data$D.freq.euc} containing the distances of documents based on \emph{maximum} metric using raw frequencies as word weights.
#' @field data$D.freq.man matrix same size as \code{data$D.freq.euc} containing the distances of documents based on \emph{manhattan} metric using raw frequencies as word weights.
#' @field data$D.freq.can matrix same size as \code{data$D.freq.euc} containing the distances of documents based on \emph{canberra} metric using raw frequencies as word weights.
#' @field data$D.freq.min matrix same size as \code{data$D.freq.euc} containing the distances of documents based on \emph{minkovsky} metric using raw frequencies as word weights.
#' @field data$D.freq.sph matrix same size as \code{data$D.freq.euc} containing the distances of documents based on \emph{spherical} metric (cosine dissimilarities) using raw frequencies as word weights.
#' @field data$D.tfidf.euc matrix similar to \code{data$D.freq.euc} contains \emph{euclidean} distances of documents using \emph{tf-idf} as word weights.
#' @field data$D.tfidf.max matrix similar to \code{data$D.freq.max} contains \emph{maximum} distances of documents using \emph{tf-idf} as word weights.
#' @field data$D.tfidf.man matrix similar to \code{data$D.freq.man} contains \emph{manhattan} distances of documents using \emph{tf-idf} as word weights.
#' @field data$D.tfidf.can matrix similar to \code{data$D.freq.can} contains \emph{canberra} distances of documents using \emph{tf-idf} as word weights.
#' @field data$D.tfidf.min matrix similar to \code{data$D.freq.min} contains \emph{minkovsky} distances of documents using \emph{tf-idf} as word weights.
#' @field data$D.tfidf.sph matrix similar to \code{data$D.freq.sph} contains \emph{spherical} distances of documents using \emph{tf-idf} as word weights.
#' @field data$CLS integer vector of size \code{Nd}.
#' Contains the cluster number associated with each text document after the clustering has been implemented.
#' @field data$CRS matrix \code{Nc x Nt} where \code{Nc} is the number of clusters and \code{Nt} is the number of terms (words).
#' Contains centers of each cluster after the clustering has been implemented.
#' @field data$CRS.dist matrix \code{Nd x Nc}.
#' Contains distances of each document from centers of each cluster based on the metric passed to method \code{centers.dist()} in its last call.
#' @field data$CNTR matrix \code{Nc x Nt} where \code{Nt} is the number of terms (words).
#' Contains centers of each cluster after the clustering has been implemented.
#' @field data$CNTR.dist vector of numerics of size \code{Nd}.
#' Contains the distaces of each document from the center of all documents using the metric passed to method \code{center.dist()} in its last call.
#'
#' @export TEXT.MINER
#' @exportClass TEXT.MINER
TEXT.MINER <- setRefClass("TEXT.MINER",
                          # https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
                          fields = list(
                            data          = "list",
                            n.text        = "integer",
                            # stop.words    = "character",
                            # dictionary    = "data.frame",
                            settings      = "list"

                            # DTM           = "matrix",
                            # W.tfidf       = "matrix",
                            # W.bin         = "matrix",
                            #
                            # D.bin         = "matrix",
                            #
                            # D.freq.euc    = "matrix",
                            # D.freq.max    = "matrix",
                            # D.freq.man    = "matrix",
                            # D.freq.can    = "matrix",
                            # D.freq.min    = "matrix",
                            # D.freq.sph    = "matrix",
                            #
                            # D.tfidf.euc   = "matrix",
                            # D.tfidf.max   = "matrix",
                            # D.tfidf.man   = "matrix",
                            # D.tfidf.can   = "matrix",
                            # D.tfidf.min   = "matrix",
                            # D.tfidf.sph   = "matrix",

                            # CLS           = "numeric",
                            # CRS           = "matrix",
                            # CRS.dist      = "matrix",
                            # CNTR            = "numeric",
                            # CNTR.dist       = "numeric"
                          ),

                          methods = list(
                            initialize = function(dataset, text_col = 'text', id_col = NULL, time_col = NULL, label_col = NULL, settings = genDefaultSettings()){
                              "
                              Class Constructor function. \n
                              \n
                              Arguments: \n
                              text_vect:  vector of character containing raw text documents. \n
                              arr_time:   vector of POSIXlt containing the time in which the text document is issued. Need to be given to the class constructor as an argument.\n
                              stop_words: vector of character specifying words to be removed from the text corpus. Default is tm::stopwords('english')
                              dictionary: data.frame of two columns containing words to be replaced by their synonyms.
                              Words in the first column are replaced by the words in the second.
                              settings:   list of various parameters containing various settings of the object.
                              Refer to the calss documentation to see all setting parameters.

                              "
                              library(gener)
                              #library(niramath)
                              support('magrittr', 'dplyr')
                              # Check Input Arguments:

                              # Prepare settings:
                              settings$remove_special_characters %<>% verify('logical', lengths = 1, domain = c(T,F), default = 'T')
                              settings$tolower %<>% verify('logical', lengths = 1, domain = c(T,F), default = 'T')
                              settings$unique %<>% verify('logical', lengths = 1, domain = c(T,F), default = 'T')
                              settings$remove_punctuation %<>% verify('logical', lengths = 1, domain = c(T,F), default = 'T')
                              settings$remove_numbers %<>% verify('logical', lengths = 1, domain = c(T,F), default = 'T')
                              settings$tm_package %<>% verify('character', lengths = 1, domain = c('tm', 'text2vec'), default = 'text2vec')
                              settings$sparsity %<>% verify('numeric', lengths = 1, domain = c(0, 1), default = 1.0)
                              settings$normalize %<>% verify('logical', lengths = 1, domain = c(T, F), default = F) # if TRUE, dtm will be normalized by rows, means rowSums(DTM) = 1, good for when documents have various length
                              settings$stop_words %<>% verify('character', default = c(tm::stopwords('english'), letters, LETTERS))
                              settings$dictionary %<>% data.frame() # todo: make it a list or provide predefined column names
                              settings$wc_gradint %<>% verify('character', lengths = 1, default = 'weight')
                              settings$prc.centralize %<>% verify('logical', domain = c(T, F), default = T)
                              settings$prc.scale %<>% verify('logical', domain = c(T, F), default = F)
                              # Todo: do it for all settings parameters

                              if (inherits(dataset, c('character', 'factor'))){
                                dataset = data.frame(ID = dataset %>% length %>% sequence %>% as.character, text = dataset %>% as.character)
                                text_col = 'text'
                                id_col   = 'ID'
                              }

                              if(is.null(id_col)){dataset$ID = dataset %>% nrow %>% sequence %>% as.character; id_col   = 'ID'}

                              dataset %<>% nameColumns(columns = list(text = text_col, ID = id_col, time = time_col, label = label_col), classes = list(ID = 'factor', text = 'character', time = 'POSIXct', label = c('integer', 'character')))
                              if (settings$unique){dataset %<>% distinct(text, .keep_all = T)}

                              # Replacement:
                              data$dataset   <<- dataset
                              settings       <<- settings
                              n.text         <<- nrow(dataset)
                            },

                            clust = function(nc = settings$num_clust, weighting = settings$weighting, metric = settings$metric){
                              "
                              Clusters the text documents on the given metric and weighting. \n
                              \n
                              Arguments: \n
                              nc:         a single integer specifying the number of clusters. \n
                              weighting:  a single character. Must be within valid.weightings\n
                              metric:     a single character. Must be within valid.metrics\n
                              Returns: integer vector containing cluster numbers associated with text documents.\n
                              "

                              W = get.weight.matrix(weighting)
                              if      (metric == 'euclidean'){
                                S = kmeans(W, nc)
                                data$CLS       <<- S$cluster
                                data$CRS       <<- S$centers
                              }
                              else if (metric == 'spherical'){
                                library(skmeans)
                                S = skmeans(W, k = nc)
                                data$CLS       <<- S$cluster
                                data$CRS       <<- S$prototypes
                              }
                              else if (metric %in% c("maximum", "manhattan", "canberra", "binary" , "minkowski")) {
                                MDS = get.mds(n.dim = min(dim(W)[1], dim(W)[2]), weighting = weighting, metric = metric)
                                S   = kmeans(MDS, nc)
                                data$CLS  <<- S$cluster
                                data$CRS  <<- S$centers}
                              else {assert(F, "Error from clust(): metric not supported!")}
                              return(data$CLS)
                            },

                            subsetObjects = function(k){
                              assert(length(k) == length(text),'Error: length of k must equal num text')

                              clusts = list()
                              for (i in 1:max(k)){
                                tmr_i  = new('TEXT.MINER', text_vect = text[k == i], settings = settings)
                                clusts = c(clusts, tmr_i)
                              }
                              return(clusts)
                            },

                            clusterObject = function(cn){
                              "
                              Returns all documents of a given cluster, as a new TEXT.MINER object. \n
                              \n
                              Arguments: \n
                              cn:         a single integer specifying the cluster number. \n
                              Returns: a fresh object of class TEXT.MINER containing only the text documents within the given cluster number.\n
                              "
                              if (is.null(data$CLS)){
                                return()
                              }
                              assert (cn <= max(data$CLS), 'Error: cn can not be greater than the number of clusters')
                              assert (length(data$CLS) == nrow(data$dataset), 'Error: count of clusters do not match count of texts')
                              return(subsetObject(data$CLS == cn))
                            },

                            clusterObjects = function(){
                              "
                              Returns each cluster as a new TEXT.MINER object. \n
                              \n
                              Arguments: \n
                              No arguments. \n
                              Returns: a list of objects of class TEXT.MINER. Each element contains the text documents within one cluster.\n
                              "
                              if (is.null(data$CLS)){
                                return()
                              }
                              return(subsetObjects(data$CLS))
                            },

                            subsetObject = function(rows){
                              tmr  = new('TEXT.MINER', dataset = data$dataset[rows,], text_col = 'text', id_col = 'ID', settings = settings)
                              return(tmr)
                            },

                            set.cluster = function(cn){
                              if (is.null(data$CLS)){
                                reset.clusters()
                              }
                              assert (cn < max(data$CLS) + 2, 'Error: cn cannot be greater than the number of clusters')

                              dev.off()
                              plot.mds.2d()

                              MDS = get.mds()
                              ss  = identify(MDS, plot = F)
                              data$CLS[ss] <<- cn

                              plot.mds.2d()
                            },

                            set.metric = function(m){
                              "
                              Changes the metric in the settings and clears all clusters. \n
                              \n
                              Arguments: \n
                              m:     a single integer specifying the metric. Must be within valid.metrics. \n
                              Returns: Norhing. Changes the metric in the settings to the given metric and clears all clusters.\n
                              "
                              assert(m %in% valid.metrics, "Error: Given metric is unknown!")
                              settings$metric <<- m
                              data$CRS       <<- NULL
                              data$CRS.dist  <<- NULL
                              data$CNTR      <<- NULL
                              data$CNTR.dist <<- NULL
                              # Todo: each metric should have it's own CLS and CRS, CLS.dist and CRS.dist
                            },

                            set.weighting = function(w){
                              assert(w %in% valid.weightings, "Error: Given weighting is unknown!")
                              settings$weighting <<- w
                              data$CRS       <<- NULL
                              data$CRS.dist  <<- NULL
                              data$CNTR      <<- NULL
                              data$CNTR.dist <<- NULL
                              # Todo: each weighting should have it's own CLS and CRS, CLS.dist and CRS.dist
                            },

                            reset.clusters = function(){
                              data$CLS <<- rep(1, nrow(data$dataset))
                            },

                            reset.settings = function(){
                              settings <<- default.settings
                            },

                            get.weight.matrix = function(weighting = settings$weighting){
                              if      (weighting == 'freq'){return(get.dtm())}
                              else if (weighting == 'tfidf'){return(get.tfidf())}
                            },

                            get.dtm = function(cn = NULL){
                              "
                              Use this method to get the document term matrix containing raw frequencies of each word in each document.
                              Arguments:
                              cn A single integer specifying the cluster number. If null(default), the whole text corpus is included.
                              Returns:
                              A numeric matrix containing the frequency of each term in each document
                              "
                              if (is.null(data$DTM)){
                                if(settings$tm_package == 'tm'){
                                  support('tm')
                                  tv  = data$dataset$text
                                  names(tv) <- data$dataset$ID
                                  crp = tv %>% VectorSource %>% Corpus

                                  ctrl = list(removePunctuation = settings$remove_punctuation,
                                              removeNumbers     = settings$remove_numbers,
                                              stopwords = length(settings$stop_words) > 0,
                                              stemming  = settings$stemming,
                                              minWordLength = 1)

                                  # if (settings$remove_punctuation){crp <- tm_map(crp, removePunctuation)}
                                  # if (settings$remove_numbers){crp <- tm_map(crp, removeNumbers)}
                                  if (settings$tolower){crp <- tm_map(crp, content_transformer(tolower))}  #convert to lower case
                                  # remove standard English stopwords, extra stopwords,
                                  if (length(settings$stop_words) != 0){crp <- tm_map(crp, removeWords, settings$stop_words)}
                                  # if (settings$plain_text){crp <- tm_map(crp, PlainTextDocument)} # make sure it's read as plain text
                                  # if (settings$stemming){
                                  #   library(SnowballC)
                                  #   crp <- tm_map(crp, stemDocument)
                                  # }
                                  # Make dictionary conversions
                                  if (inherits(settings$dictionary,'data.frame')){
                                    if (dim(settings$dictionary)[1] > 0){
                                      for (j in seq(crp)){
                                        for (i in sequence(nrow(settings$dictionary))){
                                          crp[[j]]$content <- gsub(paste0('\\<', settings$dictionary[i,1] , '\\>'), settings$dictionary[i,2], crp[[j]]$content)
                                        }
                                      }
                                    }
                                  }

                                  data$DTM <<- DocumentTermMatrix(crp, control = ctrl) %>% removeSparseTerms(settings$sparsity)  %>% as.matrix
                                  # rownames(data$DTM) <<- data$dataset$ID
                                  # data$W.bin   <<- as.matrix(weightBin(t(data$DTM)))
                                }
                                else if (settings$tm_package == 'text2vec'){
                                  funclist = list()
                                  if (settings$remove_special_characters){funclist %<>% c(remove.special.charachters)}
                                  if (settings$tolower){funclist %<>% c(tolower)}
                                  if (settings$remove_numbers){funclist %<>% c(removeNumbers)}
                                  # todo: add other modification functions
                                  textconvert = function(x) applyFunctionList(x, funclist)

                                  support('Matrix', 'data.table', 'text2vec')
                                  setDT(data$dataset)
                                  setkey(data$dataset, ID)

                                  itkn  = itoken(data$dataset$text, preprocessor = textconvert, tokenizer = word_tokenizer, ids = data$dataset$ID, progressbar = FALSE)
                                  vocab = create_vocabulary(itkn, stopwords = settings$stop_words)
                                  vectorizer = vocab_vectorizer(vocab)
                                  data$DTM <<- create_dtm(itkn, vectorizer)
                                  if(settings$normalize){data$DTM %<>% normalize("l1")} # todo: write for tm_package == 'tm' as well
                                  cls = Matrix::colSums(data$DTM > 0)
                                  data$DTM <<- data$DTM[, which(cls > (1.0 - settings$sparsity)*length(cls))]
                                }

                                if (ncol(data$DTM) == 0){
                                  data$W.tfidf <<- matrix()
                                  data$W.bin   <<- matrix()
                                  data$words   <<- character()
                                  return(DTM)
                                }

                                # Remove texts with total zero frequency:
                                r       = Matrix::rowSums(data$DTM)
                                zeros   = which(r == 0)
                                if (length(zeros) > 0){
                                  data$dataset  <<- data$dataset[- zeros,]
                                  data$DTM      <<- data$DTM[- zeros,]
                                }
                                if (dim(data$DTM)[1] == 0){
                                  data$DTM     <<- matrix()
                                  data$W.tfidf <<- matrix()
                                  data$W.bin   <<- matrix()
                                  data$words   <<- character()
                                  return(data$DTM)
                                }

                                data$words    <<- colnames(data$DTM)
                              }

                              if (is.null(cn) | is.null(data$CLS)){return (data$DTM)}
                              else {
                                DTMi = data$DTM[data$CLS == cn,]
                                freq = Matrix::colSums(DTMi)
                                DTMi = DTMi[, freq > 0]

                                return(DTMi)
                              }
                            },

                            get.lsa = function(ntopic = 3, weighting = settings$weighting){
                              vn = paste('lsa', weighting, ntopic, sep = '.')
                              if(!is.null(data[[vn]])){lsa = data[[vn]]} else {
                                lsa = LSA$new(n_topics = ntopic)
                                lsa$fit_transform(get.weight.matrix(weighting = weighting))
                                data[[vn]] <<- lsa
                              }
                              return(lsa)
                            },

                            get.doctopic = function(method = 'lda', ntopic = 3, weighting = settings$weighting){
                              if(method == 'lsa'){
                                lsa = get.lsa(ntopic = ntopic, weighting = weighting)
                                return(lsa$transform(get.weight.matrix(weighting = weighting)))
                              } else if(method == 'lda'){
                                lda = get.lda(ntopic = ntopic, weighting = weighting)
                                return(lda$transform(get.weight.matrix(weighting = weighting)))
                              }
                              else{stop('not supported!')}
                            },

                            get.topicword = function(method = 'lda', ...){
                              if(method == 'lsa'){
                                lsa = get.lsa(...)
                                return(lsa$components)
                              } else if(method == 'lda'){
                                lda = get.lda(...)
                                return(lda$components)
                              }
                              else{stop('not supported!')}
                            },

                            get.lda = function(ntopic = 3, weighting = settings$weighting){
                              vn = paste('lda', weighting, ntopic, sep = '.')
                              if(!is.null(data[[vn]])){lda = data[[vn]]} else {
                                lda = LDA$new(n_topics = ntopic, doc_topic_prior = 0.1, topic_word_prior = 0.01)
                                lda$fit_transform(x = get.weight.matrix(weighting = weighting), n_iter = 1000, convergence_tol = 0.001, n_check_convergence = 25)
                                data[[vn]] <<- lda
                              }
                              return(lda)
                            },

                            get.themes = function(nword = 5, ...){
                              tw  = get.topicword(...)
                              nws = sequence(nword)
                              words = colnames(tw)
                              return(apply(tw, 1, function(x) words[order(abs(x), decreasing = T)[nws]]))
                            },

                            get.tfidf = function(cn = NULL){
                              "
                              Returns the tf-idf weights of the document-term matrix containing tf-idf weights of each term in each document
                              Arguments:
                              cn A single integer specifying the cluster number. If null(default), the whole text corpus is included.
                              Returns:
                              A numeric matrix containing the weight of each term in each document
                              "
                              flg = is.null(cn) | is.null(data$CLS)
                              if(flg & !is.null(data$W.tfidf)){
                                return(data$W.tfidf)
                              }
                              Di = get.dtm()
                              if (!flg){
                                Di = Di[data$CLS == cn,]
                                freq = Matrix::colSums(Di)
                                Di   = Di[, freq > 0]
                              }

                              # todo: check what argument normalize does. Can it come to settings?
                              if(settings$tm_package == 'tm'){W <- Di %>% as.DocumentTermMatrix(weighting = weightTfIdf) %>% as.matrix}
                              else if(settings$tm_package == 'text2vec'){
                                tfidf = TfIdf$new()
                                W <- fit_transform(Di, tfidf)
                              }
                              if(flg){data$W.tfidf <<- W}
                              return(W)
                            },

                            get.dist = function(weighting = settings$weighting, metric = settings$metric){

                              arg.verify(weighting, metric)

                              if (metric == 'binary'){return(dist.binary())}
                              else if (weighting == 'freq'){
                                if      (metric == 'euclidean'){return(dist.freq.euclidean())}
                                else if (metric == 'maximum'){return(dist.freq.maximum())}
                                else if (metric == 'manhattan'){return(dist.freq.manhattan())}
                                else if (metric == 'canberra'){return(dist.freq.canberra())}
                                else if (metric == 'minkowski'){return(dist.freq.minkowski())}
                                else if (metric == 'spherical'){return(dist.freq.spherical())}
                                else {assert(F, "Error: Not Supported !")}
                              } else if (weighting == 'tfidf'){
                                if      (metric == 'euclidean'){return(dist.tfidf.euclidean())}
                                else if (metric == 'maximum'){return(dist.tfidf.maximum())}
                                else if (metric == 'manhattan'){return(dist.tfidf.manhattan())}
                                else if (metric == 'canberra'){return(dist.tfidf.canberra())}
                                else if (metric == 'minkowski'){return(dist.tfidf.minkowski())}
                                else if (metric == 'spherical'){return(dist.tfidf.spherical())}
                                else {assert(F, "Error: Not Supported !")}
                              }
                            },

                            get.mds  = function(n.dim = 2, weighting = settings$weighting, metric = settings$metric){
                              "
                              Multi-Dimensional Scaling is a dimensionality reduction method.
                              In this method, coordinates of text documents as vectors in the low-dimensional space
                              are computed while the sum of squares of difference in distances between all pairs of documents are minimized.
                              This method returns the equivalent vectors of text documents in a low-dimensional space using multi-dimensional scaling. \n
                              \n
                              Arguments: \n
                              n.dim:     a single integer specifying the number of dimensions of the lower-dimensional space. \n
                              weighting: a single character within valid.weightings specifying the weighting.
                              metric:    a single character within valid.metrics specifying the metric used for computing distances between text documents.
                              Returns: A matrix of numerics containing coordinates of equivalent vectors in the lower-dimensional space.\n
                              "
                              n.dim %<>% verify(c('numeric', 'integer'), lengths = 1, domain = c(2, Inf), default = 2) %>% as.integer
                              vn = paste('mds', n.dim, weighting, metric, sep = '.')
                              if(is.null(data[[vn]])){
                                data[[vn]] <<- get.dist(weighting = weighting, metric = metric) %>% cmdscale(n.dim)
                              }
                              return(data[[vn]])
                            },

                            get.prc.tfidf = function(){
                              if(is.null(data$PRC.tfidf)){
                                W = get.tfidf()
                                data$PRC.tfidf <<- prcomp(W, center = settings$prc.centralize, scale. = settings$prc.scale)
                              }
                              return(data$PRC.tfidf)
                            },

                            get.prc.freq = function(){
                              if(is.null(data$PRC.freq)){
                                W = get.dtm()
                                data$PRC.freq <<- prcomp(W, center = settings$prc.centralize, scale. = settings$prc.scale)
                              }
                              return(data$PRC.freq)
                            },

                            get.prc = function(weighting = settings$weighting){
                              switch(weighting,
                                     'freq'  = {return(get.prc.freq())},
                                     'tfidf' = {return(get.prc.tfidf())})
                              stop('weighting not recognized!')
                            },

                            plot.prc.2d = function(...){
                              PCA = get.prc(...)
                              plot(PCA$x[,1:2], col = data$CLS)
                            },

                            get.weights    = function(weighting = settings$weighting, cn = NULL){
                              "
                              Returns the vector of total term weights depending on the given weighting.
                              \n
                              Arguments: \n
                              weighting: a single character within valid.weightings specifying the weighting.
                              Returns: A named vector of numerics containing the total tf-idf or frequency weights of the terms. \n
                              "
                              verify(weighting, 'character', domain = valid.weightings, varname = 'weighting')
                              if (weighting == 'freq'){W = get.dtm(cn)} else {W = get.tfidf(cn)}
                              return(Matrix::colSums(W))
                            },

                            reset = function(){

                              # Resets the text miner object:
                              # Erases all the analysis, only the original dataset is kept: todo: consider changes in the dims of dataset due to zeros in get.dtm() method
                              data <<- list(dataset = data$dataset, n.text = nrow(data$dataset))
                            },

                            dist.binary = function(){
                              if (is.null(data$D.bin)){
                                data$D.bin <<- as.matrix(dist(get.dtm(), method = "binary"))
                              }
                              return (data$D.bin)
                            },

                            dist.freq.euclidean = function(){
                              if (is.null(data$D.freq.euc)){
                                data$D.freq.euc <<- as.matrix(dist(get.dtm(), method = "euclidean"))
                              }
                              return (data$D.freq.euc)
                            },

                            dist.freq.maximum = function(){
                              if (is.null(data$D.freq.max)){
                                data$D.freq.max <<- as.matrix(dist(get.dtm(), method = "maximum"))
                              }
                              return (data$D.freq.max)
                            },

                            dist.freq.manhattan = function(){
                              if (is.null(data$D.freq.man)){
                                data$D.freq.man <<- as.matrix(dist(get.dtm(), method = "manhattan"))
                              }
                              return (data$D.freq.man)
                            },

                            dist.freq.canberra = function(){
                              if (is.null(data$D.freq.can)){
                                data$D.freq.can <<- as.matrix(dist(get.dtm(), method = "canberra"))
                              }
                              return (data$D.freq.can)
                            },

                            dist.freq.minkowski = function(){
                              if (is.null(data$data$D.freq.min)){
                                data$D.freq.min <<- as.matrix(dist(get.dtm(), method = "minkowski"))
                              }
                              return (data$D.tfidf.min)
                            },

                            dist.jaccard = function(){
                              if (is.null(data$data$D.jac)){
                                if(tm_package == 'tm'){data$D.jac <<- as.matrix(dist(get.dtm(), method = "jaccard"))} else
                                  if(tm_package == 'text2vec'){data$D.jac <<- dist2(get.dtm(), method = 'jaccard')}
                              }
                              return (data$D.jac)
                            },

                            dist.freq.spherical = function(){
                              if (is.null(data$D.freq.sph)){
                                if(settings$tm_package == 'tm'){
                                  W.norm         = get.dtm() %>% apply(1, vect.normalize)
                                  data$D.freq.sph <<- 1 - t(W.norm) %*% W.norm
                                } else if (settings$tm_package == 'text2vec'){
                                  data$D.freq.sph <<- get.dtm() %>% dist2(method = 'cosine')
                                }
                              }
                              return (data$D.freq.sph)
                            },


                            dist.tfidf.euclidean = function(){
                              if (is.null(data$D.tfidf.euc)){
                                data$D.tfidf.euc <<- as.matrix(dist(get.tfidf(), method = "euclidean"))
                              }
                              return (data$D.tfidf.euc)
                            },

                            dist.tfidf.maximum = function(){
                              if (is.null(data$D.tfidf.max)){
                                data$D.tfidf.max <<- as.matrix(dist(get.tfidf(), method = "maximum"))
                              }
                              return (data$D.tfidf.max)
                            },

                            dist.tfidf.manhattan = function(){
                              if (is.null(data$D.tfidf.man)){
                                data$D.tfidf.man <<- as.matrix(dist(get.tfidf(), method = "manhattan"))
                              }
                              return (data$D.tfidf.man)
                            },

                            dist.tfidf.canberra = function(){
                              if (is.null(data$D.tfidf.can)){
                                data$D.tfidf.can <<- as.matrix(dist(get.tfidf(), method = "canberra"))
                              }
                              return (data$D.tfidf.can)
                            },

                            dist.tfidf.minkowski = function(){
                              if (is.null(data$D.tfidf.min)){
                                data$D.tfidf.min <<- as.matrix(dist(get.tfidf(), method = "minkowski"))
                              }
                              return (data$D.tfidf.min)
                            },

                            dist.tfidf.spherical = function(){
                              if (is.null(data$D.tfidf.sph)){
                                W.norm         = apply(get.tfidf(), 1, vect.normalize)
                                data$D.tfidf.sph <<- 1 - t(W.norm) %*% W.norm
                              }
                              return (data$D.tfidf.sph)
                            },

                            get.words      = function(){
                              if (is.null(data$words)){
                                D = get.dtm()
                                data$words <<- colnames(D)
                              }
                              return(data$words)
                            },

                            frequent.words = function(freq_threshold = 20){
                              f = word.freq()
                              f = f[f > freq_threshold]
                              return(names(sort(f, decreasing = TRUE)))
                            },

                            word.freq       = function(){return(Matrix::colSums(get.dtm()))},

                            term.weights    = function(){
                              "
                              Returns term weights as a data.frame of two columns.
                              The first column contains raw frequencies and the second column, contains tf-idf weights of the word in each corpus.
                              Words appear as rownames of the data.frame
                              "
                              v1 = get.weights(weighting = 'freq')
                              v2 = get.weights(weighting = 'tfidf')
                              df = data.frame( Frequency = v1, TFIDF = v2)
                              # rownames(df) <- get.words()
                              return(df)
                            },

                            plot.wordCloud = function(weighting = settings$weighting, package = 'wordcloud', cn = NULL, ...){
                              # Verifications:
                              verify(package, 'character', domain = c('wordcloud', 'wordcloud2'), varname = 'package')
                              assert(require(package, character.only = T), "Package " %++% package %++% "is not installed!", err_src = match.call()[[1]])
                              verify(cn, c('integer', 'numeric'), domain = c(1,max(data$CLS)), varname = 'cn')

                              v        = sort(get.weights(weighting = weighting, cn = cn), decreasing = T)
                              rnd.cols = F
                              cols     = settings$wc_color
                              if (settings$wc_gradient %in% c('weight', 'random')){
                                if (settings$wc_gradient == 'random'){vp = runif(length(v))} else {vp = sort(get.weights(weighting = weighting, cn = cn), decreasing = T)}
                                col      = round(vect.map(vp, 1, 9))
                                pallete  = brewer.pal(9, plural.col(settings$wc_color)) # blue gradient
                                cols     = pallete[col]
                              } else if (settings$wc_gradient == 'none'){cols = rep(settings$wc_color, length(v))}
                              else {assert(F, "Error: Not Supported!")}
                              switch(package,
                                     'wordcloud' = {
                                       wordcloud(
                                         names(v), v, min.freq = quantile(v)[2], max.words = settings$wc_max_words,
                                         random.order = F, random.color = rnd.cols, rot.per = settings$wc_rot_per,
                                         ordered.colors = T, colors = cols)},
                                     'wordcloud2' = {
                                       ww = data.frame(word = names(v), freq = v)
                                       wordcloud2(data = ww, rotateRatio = settings$wc_rot_per, ...)}
                              )
                            },

                            plot.topics = function(...){
                              support('LDAvis')
                              lda = get.lda(...)
                              lda$plot()
                            },

                            plot.mds.2d    = function(weighting = settings$weighting, metric = settings$metric, plotter = 'graphics'){
                              # Verifications:
                              verify(plotter, 'character', lengths = 1, domain = c('graphics', 'rCharts'))
                              assert(require(plotter, character.only = T), "Package " %++% package %++% "is not installed!")

                              MDS = get.mds(n.dim = 2, weighting = weighting, metric = metric)
                              if (is.null(data$CLS)){clrs = settings$plot_color} else {clrs = data$CLS}
                              switch(plotter,
                                     'graphics' = {plot(MDS, col = clrs)},
                                     'rCharts' = {
                                       MDS = cbind(MDS, as.factor(clrs))
                                       colnames(MDS) <- c('DIM.1', 'DIM.2', 'Cluster')
                                       rownames(MDS) <- paste('Document', sequence(nrow(MDS)))
                                       rCharts.scatter.plot(MDS, x = 'DIM.1', y = 'DIM.2', color = 'Cluster')
                                     })
                            },

                            plot.mds.3d    = function(weighting = settings$weighting, metric = settings$metric){
                              library(rgl)

                              MDS = get.mds(n.dim = 3, weighting = weighting, metric = metric)
                              if (is.null(data$CLS)){clrs = settings$plot_color} else {clrs = data$CLS}
                              plot3d(MDS, col = clrs)
                            },

                            plot.clusters = function(nc = settings$num_clust, weighting = settings$weighting, metric = settings$metric){
                              library(cluster)
                              W = get.weight.matrix(weighting = weighting)
                              if (dim(W)[1] < dim(W)[2]){
                                W = W[, order(word.freq(), decreasing = T)[1:(dim(W)[1])]]
                              }
                              if (is.null(data$CLS)){get.clusters(nc, weighting, metric)}
                              clusplot(W, data$CLS, color=T, shade=T, labels=2, lines=0, cex=0.7)
                            },

                            plot.wordBar   = function(weighting = settings$weighting){
                              weight <- sort(get.weights(weighting), decreasing=TRUE)
                              # Plot word frequencies
                              wf <- data.frame(word=names(weight), weight=weight)

                              library(ggplot2)
                              thr <- quantile(weight)[4]
                              p <- ggplot(subset(wf, weight > thr), aes(word, weight))
                              p <- p + geom_bar(stat="identity")
                              p <- p + theme(axis.text.x=element_text(angle=45, hjust=1))
                              p                                  },

                            # These are three functions that return data.frames:
                            # table.clusters
                            # table.words
                            # table.texts
                            table.clusters = function(){

                            },

                            word.dist  = function(weighting = settings$weighting, metric = settings$metric){
                              W = t(get.weight.matrix(weighting = weighting))
                              if      (metric == 'binary')   {return(as.matrix(dist(W, method = 'binary')))}
                              else if (metric == 'euclidean'){return(as.matrix(dist(W, method = 'euclidean')))}
                              else if (metric == 'spherical'){
                                W.norm         = apply(W, 1, vect.normalize)
                                return(1 - t(W.norm) %*% W.norm)
                              }
                              else {assert(F, "Error: Not Supported !")}
                            },

                            center     = function(weighting = settings$weighting, metric = settings$metric){
                              W = get.weight.matrix(weighting = weighting)
                              if (metric != 'euclidean'){W = matrix.normalize(W, 2)}
                              data$CNTR <<- colMeans(W)
                              return(data$CNTR)
                            },

                            centers     = function(weighting = settings$weighting, metric = settings$metric){
                              W  = get.weight.matrix(weighting = weighting)
                              if (is.null(data$CLS)){return()}
                              nc = max(data$CLS)
                              if (nc == 0){return()}
                              nw = dim(W)[2]
                              data$CRS <<- zeros(nc, nw)
                              for (i in 1:nc){
                                Wi = W[data$CLS == i,]
                                if (metric != 'euclidean'){Wi = matrix.normalize(Wi, 2)}
                                data$CRS[i,] <<- colMeans(Wi)
                              }
                              colnames(data$CRS) <<- words()
                              return(data$CRS)
                            },

                            # Returns distance of each text to the center of each cluster
                            centers.dist = function(weighting = settings$weighting, metric = settings$metric){
                              if (is.null(data$CRS)){centers()}
                              W  = get.weight.matrix(weighting = weighting)
                              nc = dim(data$CRS)[1]
                              nd = dim(W)[1]
                              data$CRS.dist <<- zeros(nd, nc)
                              for (i in sequence(nd)){
                                for (j in sequence(nc)){
                                  data$CRS.dist[i, j] <<- vect.dist(W[i,], data$CRS[j,], metric = metric)
                                }
                              }
                              return(data$CRS.dist)
                            },
                            center.dist = function(weighting = settings$weighting, metric = settings$metric){
                              if (is.null(data$CNTR)){center()}
                              W  = get.weight.matrix(weighting = weighting)
                              nd = dim(W)[1]
                              data$CNTR.dist <<- rep(0, nd)
                              for (i in 1:nd){
                                data$CNTR.dist[i] <<- vect.dist(W[i,], data$CNTR)
                              }
                              return(data$CNTR.dist)
                            }

                          ))

# Some generic functions

setMethod("length", "TEXT.MINER", function(x) nrow(x$data$dataset))

# length is a pre-defined generic function like:
# summary, plot, show, print, ...
#
# you can define your own generic function name like
# face(x)

setGeneric("words", function(x) standardGeneric('words'))

setMethod("words", "TEXT.MINER", function(x) {
  if (is.null(x$data$words)){D = x$get.dtm()}
  return(x$data$words)
})


# define a setter:
setGeneric("metric<-", function(x, value) standardGeneric("metric<-"))
setReplaceMethod("metric", "TEXT.MINER", function(x, value) {
  assert(value %in% valid.metrics, "Error: Given metric is unknown!")
  x$settings$metric <<- value
  x$data$CRS       <<- NULL
  x$data$CRS.dist  <<- NULL
  x$data$CNTR        <<- NULL
  x$data$CNTR.dist   <<- NULL
  x
})

# define the validity method:
# setValidity("TEXT.MINER", function(object){
#   if !is.character(objects$settings$metric)
# })



# Todo:
#  1- similarly change function set.weighting() to a setter method
#  2- write a setValidity() method
#  3- add a coercion method


