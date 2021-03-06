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
    #images.test[index,] <- data.frame(t(r),t(g),t(b),c(l+1))
  }
  close(to.read)
  remove(l,r,g,b,f,i,index, to.read)
}
length(images.rgb)
img.col.mat<-list()
#Converts images to grayscale
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
#grayscale images are combined with the labels
final<-as.data.frame(images.gray)
colnames(final)<-c(colnames(final[,1:1024]),"label")
final$label<-as.factor(final$label)
#QDA requires MASS package
install.packages("MASS")
library(MASS)
set.seed(1)
#Separating test and train data
data <- final
index <- sample(1:nrow(data),round(.8 * nrow(data)))
qda.fit <- qda(data$label~.,data = data, subset = index)
qda.fit
#model predicted on the test data
qda.pred <- predict(qda.fit,data[-index,])
qda.class <- qda.pred$class
table(qda.class,data[-index,'label'])
mean(qda.class == data[-index,'label'])