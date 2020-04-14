library(boot)
library(gbm)
library(ggplot2)
library(mgcv)
library(e1071)
library(randomForest)
#
#Dataset Preparation
math = read.csv("C:\\Users\\Susana\\Documents\\Universidad\\Machine Learning\\Coursework\\student-mat.csv", sep=";", header=TRUE)
port = read.csv("C:\\Users\\Susana\\Documents\\Universidad\\Machine Learning\\Coursework\\student-por.csv", sep=";", header=TRUE)
#Searching for nulls
sum(is.na(math),is.na(port))
#Combining both datasets
math = math[,c(1:30,33)]
port = port[,c(1:30,33)]
category = rep("math", nrow(math))
math_new = data.frame(category, math)
math_new[1:5,1:10]
category = rep("port", nrow(port))
port_new = data.frame(category, port)
port_new[1:5,1:10]
mp = rbind(math_new, port_new)
summary(mp)
#Dividing in two datasets for training and testing
#combined dataset
set.seed(0)
sampling = sample(nrow(mp), (nrow(mp)/3)*2)
mp.train = mp[sampling,]
mp.test = mp[-sampling, 1:31]
mp.test.G3 = mp[-sampling, 32]
#math
set.seed(0)
sampling.math = sample(nrow(math), (nrow(math)/3)*2)
math.train = math[sampling.math,]
math.test = math[-sampling.math, 1:30]
math.test.G3 = math[-sampling.math, 31]
#portuguese
set.seed(0)
sampling.port = sample(nrow(port), (nrow(port)/3)*2)
port.train = port[sampling.port,]
port.test = port[-sampling.port, 1:30]
port.test.G3 = port[-sampling.port, 31]
#
#Tree-Based with Boosting
#combined
boost = gbm(G3~., data=mp.train, distribution="gaussian", shrinkage=0.00000001, n.trees=5000, interaction.depth=4)
summary(boost)
boost.pred = predict(boost, newdata=mp.test, n.trees=5000)
mean.tree = mean((boost.pred - mp.test.G3)^2)
mean.tree
tree.acc = mean(round(boost.pred) == mp.test.G3)*100
tree.acc
#math
boost.math = gbm(G3~., data=math.train, distribution="gaussian", shrinkage=0.00000001, n.trees=5000, interaction.depth=4)
summary(boost.math)
boost.pred.m = predict(boost.math, newdata=math.test, n.trees=5000)
mean.tree.m = mean((boost.pred.m - math.test.G3)^2)
mean.tree.m
tree.acc.m = mean(round(boost.pred.m) == math.test.G3)*100
tree.acc.m
#portuguese
boost.port = gbm(G3~., data=port.train, distribution="gaussian", shrinkage=0.00000001, n.trees=5000, interaction.depth=4)
summary(boost.port)
boost.pred.p = predict(boost.port, newdata=port.test, n.trees=5000)
mean.tree.p = mean((boost.pred.p - port.test.G3)^2)
mean.tree.p
tree.acc.p = mean(round(boost.pred.p) == port.test.G3)*100
tree.acc.p
#
#Random Forests
#combined
set.seed(5)
forests = randomForest(G3~., data=mp.train, importance=TRUE, ntree=1000)
forests.pred = predict(forests, newdata=mp.test)
forests
importance(forests)
varImpPlot(forests, n.var=10, main="Random Forests", pch=16)
mean.randomf = mean((forests.pred-mp.test.G3)^2)
mean.randomf
f.acc = mean(round(forests.pred) == mp.test.G3)*100
f.acc
#math
set.seed(5)
forests.m = randomForest(G3~., data=math.train, importance=TRUE, ntree=1000)
forests.pred.m = predict(forests.m, newdata=math.test)
importance(forests.m)
varImpPlot(forests.m, n.var=10, main="", pch=16)
mtext("Math Random Forest", cex=1.2, adj = -0.1)
mean.randomf.m = mean((forests.pred.m-math.test.G3)^2)
mean.randomf.m
f.acc.m = mean(round(forests.pred.m) == math.test.G3)*100
f.acc.m
#portuguese
set.seed(5)
forests.p = randomForest(G3~., data=port.train, importance=TRUE, ntree=1000)
forests.pred.p = predict(forests.p, newdata=port.test)
importance(forests.p)
varImpPlot(forests.p, n.var=10, pch=16, main="")
mtext("Portuguese Random Forest", cex=1.2, adj = -1.25)
mean.randomf.p = mean((forests.pred.p-port.test.G3)^2)
mean.randomf.p
f.acc.p = mean(round(forests.pred.p) == port.test.G3)*100
f.acc.p
#
#SVM
#Support Vector Machine
#combined linear
set.seed(6)
tune.out = tune(svm, G3~., data=mp.train, kernel="linear", cost=1)
best.linear = tune.out$best.model
ypred.linear = predict(best.linear, mp.test)
mean.svml = mean((ypred.linear-mp.test.G3)^2)
mean.svml
svm.linear.acc = mean(round(ypred.linear) == mp.test.G3)*100
svm.linear.acc
#math linear
set.seed(6)
tune.out.m = tune(svm, G3~., data=math.train, kernel="linear", cost=1)
best.linear.m = tune.out.m$best.model
ypred.linear.m = predict(best.linear.m, math.test)
mean.svml.m = mean((ypred.linear.m-math.test.G3)^2)
mean.svml.m
svm.linear.acc.m = mean(round(ypred.linear.m) == math.test.G3)*100
svm.linear.acc.m
#portuguese linear
set.seed(6)
tune.out.p = tune(svm, G3~., data=port.train, kernel="linear", cost=1)
best.linear.p = tune.out.p$best.model
ypred.linear.p = predict(best.linear.p, port.test)
mean.svml.p = mean((ypred.linear.p-port.test.G3)^2)
mean.svml.p
svm.linear.acc.p = mean(round(ypred.linear.p) == port.test.G3)*100
svm.linear.acc.p
#combined polynomial of 2nd degree
set.seed(7)
tune.out.p = tune(svm, G3~., data=mp.train, kernel="polynomial", cost=1, degree=2)
ypred.p = predict(tune.out.p$best.model, mp.test)
svm.p.acc = mean(round(ypred.p) == mp.test.G3)*100
svm.p.acc
mean.svmp = mean((ypred.p-mp.test.G3)^2)
mean.svmp
#math polynomial of 2nd degree
set.seed(6)
tune.out.pm = tune(svm, G3~., data=math.train, kernel="polynomial", degree=2, cost=1)
ypred.linear.pm = predict(tune.out.pm$best.model, math.test)
mean.svml.pm = mean((ypred.linear.pm-math.test.G3)^2)
mean.svml.pm
svm.acc.pm = mean(round(ypred.linear.pm) == math.test.G3)*100
svm.acc.pm
#portuguese polynomial of 2nd degree
set.seed(6)
tune.out.pp = tune(svm, G3~., data=port.train, kernel="linear", cost=1, degree=2)
ypred.linear.pp = predict(tune.out.pp$best.model, port.test)
mean.svml.pp = mean((ypred.linear.pp-port.test.G3)^2)
mean.svml.pp
svm.acc.pp = mean(round(ypred.linear.pp) == port.test.G3)*100
svm.acc.pp
#combined polynomial of 3rd degree
set.seed(8)
tune.out.p3 = tune(svm, G3~., data=mp.train, kernel="polynomial", cost=1, degree=3)
ypred.p3 = predict(tune.out.p3$best.model, mp.test)
svm.p.acc3 = mean(round(ypred.p3) == mp.test.G3)*100
svm.p.acc3
mean.svmp3 = mean((ypred.p3-mp.test.G3)^2)
mean.svmp3
#math polynomial of 3rd degree
set.seed(6)
tune.out.p3m = tune(svm, G3~., data=math.train, kernel="polynomial", degree=3, cost=1)
ypred.linear.p3m = predict(tune.out.p3m$best.model, math.test)
mean.svml.p3m = mean((ypred.linear.p3m-math.test.G3)^2)
mean.svml.p3m
svm.acc.p3m = mean(round(ypred.linear.p3m) == math.test.G3)*100
svm.acc.p3m
#portuguese polynomial of 3rd degree
set.seed(6)
tune.out.p3p = tune(svm, G3~., data=port.train, kernel="linear", cost=1, degree=3)
ypred.linear.p3p = predict(tune.out.p3p$best.model, port.test)
mean.svml.p3p = mean((ypred.linear.p3p-port.test.G3)^2)
mean.svml.p3p
svm.acc.p3p = mean(round(ypred.linear.p3p) == port.test.G3)*100
svm.acc.p3p
#
#Models Comparison
#combined
barplot(c(mean.tree, mean.randomf, mean.svml, mean.svmp, mean.svmp3),
        xlab="Models",
        ylab="Test MSEs",
        main="MSE Comparison",
        names.arg=c("Tree-B","R. Forest", "SMV L", "SMV P2", "SMV P3"),
        col="purple")
barplot(c(tree.acc, f.acc, svm.linear.acc, svm.p.acc, svm.p.acc3),
        xlab="Models",
        ylab="Accuracy %",
        main="Accuracy Comparison",
        names.arg=c("Tree-B","R. Forest", "SVM L", "SVM P2", "SVM P3"),
        col="orange")
#math
barplot(c(mean.tree.m, mean.randomf.m, mean.svml.m, mean.svml.pm, mean.svml.p3m),
        xlab="Models",
        ylab="Test MSEs",
        main="MSE Comparison for Math",
        names.arg=c("Tree-B","R. Forest", "SMV L", "SMV P2", "SMV P3"),
        col="purple")
barplot(c(tree.acc.m, f.acc.m, svm.linear.acc.m, svm.acc.pm, svm.acc.p3m),
        xlab="Models",
        ylab="Accuracy %",
        main="Accuracy Comparison for Math",
        names.arg=c("Tree-B","R. Forest", "SVM L", "SVM P2", "SVM P3"),
        col="orange")
#portuguese
barplot(c(mean.tree.p, mean.randomf.p, mean.svml.p, mean.svml.pp, mean.svml.p3p),
        xlab="Models",
        ylab="Test MSEs",
        main="MSE Comparison for Portuguese",
        names.arg=c("Tree-B","R. Forest", "SMV L", "SMV P2", "SMV P3"),
        col="purple")
barplot(c(tree.acc.p, f.acc.p, svm.linear.acc.p, svm.acc.pp, svm.acc.p3p),
        xlab="Models",
        ylab="Accuracy %",
        main="Accuracy Comparison for Portuguese",
        names.arg=c("Tree-B","R. Forest", "SVM L", "SVM P2", "SVM P3"),
        col="orange")
