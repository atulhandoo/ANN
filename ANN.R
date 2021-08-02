library(h2o)
library(jpeg)
library(caTools)
#hyp<-matrix(nrow = 1500, ncol = 5237)
hyp<-matrix(nrow = 1500, ncol = 69)
filenames <- list.files(path = ".")
for(i in 1 : length(filenames)){
hyp[i,1]<-filenames[i]
file<-readJPEG(filenames[i])
file1<-c(file)
file1<-colMeans(file)
#hyp[i,2:5237]<-round((file1*100)^2,0)
hyp[i,2:69]<-round((file1*100)^2,0)
cat(i,'\n')
}

hyp<-as.data.frame(hyp)
#hyp[,2:5237]<-as.numeric(unlist(hyp[,2:5237]))
hyp[,2:69]<-as.numeric(unlist(hyp[,2:69]))
#hyp<-cbind(hyp[,1], abs(hyp[,3:5237] - hyp[,2:5236]))
hyp<-cbind(hyp[,1], abs(hyp[,3:69] - hyp[,2:68]))
#hyp[,2:5236][hyp[,2:5236]<20]<-0
hyp[,2:68][hyp[,2:68]<20]<-0
hyp$label<-as.numeric(substr(hyp[,1],6,7))
hyp$label<-as.factor(hyp$label)
hist(as.numeric(unlist(c(hyp[,2:68]))),xlim = c(0,500))
summary(as.numeric(unlist(c(hyp[,2:68]))))
boxplot(as.numeric(unlist(c(hyp[,2:68]))))


h2o.no_progress()
h2o.init(nthreads=3)
#h2o.shutdown()
sample<-sample.split(hyp$label,SplitRatio = 0.80)
train_hyp <- subset(hyp,sample ==TRUE)
test<- subset(hyp,sample ==FALSE)
sample<-sample.split(train_hyp$label,SplitRatio = 0.85)
train<-subset(train_hyp,sample ==TRUE)
valid<-subset(train_hyp,sample ==FALSE)
str(train$V3)

train_h2o<-as.h2o(train)
valid_h2o <- as.h2o(valid)
test_h2o <- as.h2o(test)


# Selecting hyper-parameters:
#hyper_params <- list(activation = c("Rectifier","Tanh","Maxout", "RectifierWithDropout","TanhWithDropout", "MaxoutWithDropout"), 
#                     hidden = list(c(5000, 3000, 500, 50), c(3000, 500, 50), c(500, 50)), input_dropout_ratio = c(0, 0.05, 0.1), 
#                     l1 = seq(0, 1e-5, 1e-4), l2 = seq(0, 1e-5, 1e-4))

hyper_params <- list(activation = c("Rectifier","Tanh","Maxout", "RectifierWithDropout","TanhWithDropout", "MaxoutWithDropout"), 
                    hidden = list(c(67, 65, 61, 59), c(65, 61, 59), c(61, 59)), input_dropout_ratio = c(0, 0.05, 0.1), 
                     l1 = seq(0, 1e-5, 1e-4), l2 = seq(0, 1e-5, 1e-4))

# Selecting optimal model search criteria. Search will stop once top 5 models are within 1% of each other:
search_criteria = list(strategy = "RandomDiscrete", max_runtime_secs = 1000, max_models = 500, seed=123, 
                      stopping_rounds=10, stopping_tolerance=0.01)

y<-"label"
x<-setdiff(names(train),c(y,"hyp[,1]","img"))

# Running search for the optimal models:
dl_random_grid <- h2o.grid(algorithm="deeplearning", grid_id = "dl_grid_random", training_frame=train_h2o, 
                           validation_frame=valid_h2o, x=x, y=y, epochs=10, stopping_metric="logloss", 
                           stopping_tolerance=0.01, stopping_rounds=3, hyper_params = hyper_params, 
                           search_criteria = search_criteria)    


# Sorting models:                            
grid <- h2o.getGrid("dl_grid_random",sort_by="logloss", decreasing=FALSE)
summary(dl_random_grid, show_stack_traces = TRUE)

h2o.no_progress()
grid@summary_table[1,]

best_model <- h2o.getModel(grid@model_ids[[1]]) 
best_model

yhat <- h2o.predict(best_model, test_h2o)
cm<-h2o.confusionMatrix(best_model, test_h2o)
(1-cm[31,31])*100

h2o.no_progress()
# Model 2
grid@summary_table[1,]

# Model 3
grid@summary_table[3,]

