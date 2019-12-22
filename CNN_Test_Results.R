labels <- read.table("batches.meta.txt")
images.rgb <- list()
images.lab <- list()
num.images = 10000 # Set to 10000 to retrieve all images per file to memory
# Cycle through all 5 binary files
for (f in 1:5) {
  to.read <- file(paste("data_batch_", f, ".bin", sep=""), "rb")
  for(i in 1:num.images) {
    l <- readBin(to.read, integer(), size=1, n=1, endian="big")
    r <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
    g <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
    b <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
    index <- num.images * (f-1) + i
    images.rgb[[index]] = data.frame(r, g, b)
    images.lab[[index]] = l+1
  }
  close(to.read)
  remove(l,r,g,b,f,i,index, to.read)
}
# function to run sanity check on photos & labels import
drawImage <- function(index) {
  # Testing the parsing: Convert each color layer into a matrix,
  # combine into an rgb object, and display as a plot
  img <- images.rgb[[index]]
  img.r.mat <- matrix(img$r, ncol=32, byrow = TRUE)
  img.g.mat <- matrix(img$g, ncol=32, byrow = TRUE)
  img.b.mat <- matrix(img$b, ncol=32, byrow = TRUE)
  img.col.mat <- rgb(img.r.mat, img.g.mat, img.b.mat, maxColorValue = 255)
  dim(img.col.mat) <- dim(img.r.mat)
  # Plot and output label
  library(grid)
  grid.raster(img.col.mat, interpolate=FALSE)
  # clean up
  remove(img, img.r.mat, img.g.mat, img.b.mat, img.col.mat)
  labels[[1]][images.lab[[index]]]
}
drawImage(sample(1:(num.images*5), size=1))
#Test_batch
labels1 <- read.table("batches.meta.txt")
images.rgb1.test<- list()
images.lab.test<- list()
num.images1=10000
to.read <- file(paste("test_batch",".bin", sep=""),"rb")
for(i in 1:num.images1) {
  l <- readBin(to.read, integer(), size=1, n=1, endian="big")
  r <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
  g <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
  b <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
  images.rgb1.test[[i]]<- data.frame(r,g,b)
  images.lab.test[[i]]<- l+1
}
close(to.read)
remove(l,r,g,b,i, to.read)
data <- images.rgb
data.test<- images.rgb1.test
#Creating an array of 32x32x3 for 50000 images to give it as input in the CNN, as
#it takes data in the form of height, width and depth
data_ <- array(dim = c(length(data),32,32,3))
for (i in 1:length(data)){
  d_1 <- matrix(data[[i]][,1],nrow=32,ncol=32,byrow = TRUE)
  for (j in 1:32){
    data_[i,j,,1] <-as.numeric(unlist(d_1[j,]))
  }
  d_2 <- matrix(data[[i]][,2],nrow=32,ncol=32,byrow = TRUE)
  for (j in 1:32){
    data_[i,j,,2] <-as.numeric(unlist(d_2[j,]))
  }
  d_3 <- matrix(data[[i]][,3],nrow=32,ncol=32,byrow = TRUE)
  for (j in 1:32){
    data_[i,j,,3] <-as.numeric(unlist(d_3[j,]))
  }
  rm(d_1,d_2,d_3)
}
data_test <- array(dim = c(length(data.test),32,32,3))
for (i in 1:length(data.test)){
  d_1 <- matrix(data.test[[i]][,1],nrow=32,ncol=32,byrow = TRUE)
  for (j in 1:32){
    data_test[i,j,,1] <-as.numeric(unlist(d_1[j,]))
  }
  d_2 <- matrix(data.test[[i]][,2],nrow=32,ncol=32,byrow = TRUE)
  for (j in 1:32){
    data_test[i,j,,2] <-as.numeric(unlist(d_2[j,]))
  }
  d_3 <- matrix(data.test[[i]][,3],nrow=32,ncol=32,byrow = TRUE)
  for (j in 1:32){
    data_test[i,j,,3] <-as.numeric(unlist(d_3[j,]))
  }
  rm(d_1,d_2,d_3)
}
#Separating the dataset for training and testing
set.seed(02122019)
train_x <- data_[1:50000,,,]
test_x <- data_test[1:10000,,,]
train_y <- array(unlist(images.lab[1:50000]), dim = c(50000,1))
test_y <- array(unlist(images.lab.test[1:10000]), dim = c(10000,1))
#labels are in the range of 0 to 9,so we are subtracting a value of 1
#If its not done, there could be an error like index out of bound
train_y1 <- train_y - 1
test_y1 <- test_y - 1
install.packages("reticulate")
library(reticulate)
#Devtools is used to download files from github and from that keras package is downloaded.
install.packages("devtools")
library(devtools)
devtools::install_github("rstudio/keras")
force= TRUE
library(keras)
#install_tensorflow(gpu = T) uses GPU for computation
install.packages("tensorflow")
library(tensorflow)
install_tensorflow(gpu=T)
#One Hot Encoding
train_y2 <- to_categorical(train_y1,num_classes = 10)
test_y2 <- to_categorical(test_y1,num_classes = 10)
model<-keras_model_sequential()
#Developing an architecture for the CNN
#Relu performs element wise operation and locates the features
model %>%
  layer_conv_2d(filter=32,kernel_size=c(3,3),padding="same",input_shape=c(32,32,3),activation = "relu")%>%layer_conv_2d(filter=32 ,kernel_size=c(3,3),activation = "relu")%>%layer_max_pooling_2d(pool_size=c(2,2))%>%layer_dropout(0.25) %>%layer_conv_2d(filter=32,kernel_size=c(3,3),padding="same",activation = "relu")%>%layer_conv_2d(filter=32,kernel_size=c(3,3),activation = "relu" )%>%layer_max_pooling_2d(pool_size=c(2,2)) %>%layer_dropout(0.25) %>%layer_flatten() %>%layer_dense(units = 512,activation = "relu")%>%layer_dropout(0.5)%>%layer_dense(10) %>% #As we have 10 labels so the output is
  layer_activation("softmax")#Popular optimizer Adam is used with a learning rate of 0.0001
opt<-optimizer_adam( lr= 0.0001 , decay = 1e-6 )
#Categorical Crossentropy compiles the accuracy of classification
model%>%compile(loss="categorical_crossentropy",optimizer=opt,metrics = "accuracy")
summary(model)
#Model is fit on the train and test data along with their labels with an iteration of 80
history<- model%>%fit(train_x,train_y2,batch_size=32,epochs=80,validation_data = list(test_x, test_y2),shuffle=TRUE)
plot(history)
eval<- model %>% evaluate(test_x,test_y2)