url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
crx <- read.table( file=url, header=FALSE, sep="," )


write.table( crx, "crx.dat", quote=FALSE, sep="," )

write.table( crx, "crx.dat", quote=FALSE, sep="," )

head( crx, 6 )

crx <- crx[ sample( nrow( crx ) ), ]
X <- crx[,1:15]
y <- crx[,16]


trainX <- X[1:600,]
trainy <- y[1:600]
testX <- X[601:690,]
testy <- y[601:690]

library(C50)
trainy<-as.factor(trainy)

model <- C50::C5.0( trainy ~ ., trainX )
summary( model)

#Boosting
#Choosing 10 trails is kind of de-facto.
model <-  C50::C5.0( trainX, trainy, trials=10 )
model

p <- predict( model, testX, type="class" )



