labels <- read.table("batches.meta.txt")
images.rgb <- list()
images.lab <- list()
#images.test <- data.frame(matrix(vector(),0,3073))
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
    #images.test[index,] <- data.frame(t(r),t(g),t(b),c(l+1))
  }
  close(to.read)
  remove(l,r,g,b,f,i,index, to.read)
}
length(images.rgb)
img.col.mat<-list()
########### Converts images to grayscale
images.gray <- matrix(nrow=50000,ncol=1025)
for(index in 1:50000){
  img <- images.rgb[[index]]
  img.r.mat <- matrix(img$r, byrow = TRUE)
  img.g.mat <- matrix(img$g, byrow = TRUE)
  img.b.mat <- matrix(img$b, byrow = TRUE)
  img.gray.mat <- (img.r.mat + img.g.mat + img.b.mat)/3
  images.gray[index,]<-c(img.gray.mat,images.lab[[index]])
  rm(img,img.r.mat,img.b.mat,img.g.mat,img.gray.mat)
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
drawImageGrayscale <- function(index){
  img <- images.rgb[[index]]
  img.r.mat <- matrix(img$r, ncol=32, byrow = TRUE)
  img.g.mat <- matrix(img$g, ncol=32, byrow = TRUE)
  img.b.mat <- matrix(img$b, ncol=32, byrow = TRUE)
  img.col.mat <- (img.b.mat + img.g.mat + img.r.mat)/3
  img.col.mat <- rgb(img.col.mat,img.col.mat,img.col.mat,maxColorValue = 255)
  dim(img.col.mat) <- dim(img.r.mat)
  library(grid)
  grid.raster(img.col.mat, interpolate=FALSE)
  # clean up
  remove(img, img.r.mat, img.g.mat, img.b.mat, img.col.mat)
  labels[[1]][images.lab[[index]]]
}
drawImage(2)
drawImageGrayscale(2)
####Upto this we have worked with grayscale
####now we should try pca and then move to randomForest
#
final<-as.data.frame(images.gray)
colnames(final)<-c(colnames(final[,1:1024]),"label")
final$label<-as.factor(final$label)

##Without PCA but with grayscale
train<- sample(50000,45000)
trainx<- final[train,]
testx<- final[-train,]
tree.img<- randomForest(label~.,data = trainx,mtry=32,ntree=20,importance=TRUE)
summary(tree.img)
pred<- predict(tree.img,newdata = testx)
table(pred,testx$label)
mean(pred==testx$label)
#No Pca and grayscale

#PCA with grayscale
install.packages("e1071")
library(e1071)
pca<- prcomp(final[1:50000,1:1024])
pc.df<- data.frame(pca$x)
#To get the scree plot
screep<- function(x)
{var<- x$sdev^2
pvar<- var/sum(var)
print("Proportion of Variance")
print(pvar)
par(mfrow=c(2,2))
plot(pvar,xlab = "Principal Components",ylab = "Proportion of Variance explained",ylim = c(0,1),type = "b")
plot(cumsum(pvar),xlab = "Principal Components",ylab = "Cumulative Proportion of Variance",ylim = c(0,1),type = "b")
screeplot(x)
screeplot(x,type = "l")
par(mfrow=c(1,1))}
screep(pca)
lab1<- final$label[1:50000]
pca_d<- cbind.data.frame(pc.df,lab1)
#Selecting the Important Principal Components
training<- pca_d[,1:200]
training<- cbind(training,lab1)
#Separating for training and test data
train<- sample(50000,45000)
trainx<- training[train,]
testx<- training[-train,]
#Random Forest on pca and grayscale
install.packages("randomForest")
library(randomForest)
tree.img<- randomForest(lab1~.,data = trainx,mtry=32,ntree=64,importance=TRUE)
summary(tree.img)
pred<- predict(tree.img,newdata = testx)
table(pred,testx$lab1)
mean(pred==testx$lab1)