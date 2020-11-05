% Implements the framework presented in
% "Identifying oil spill types based on remotely sensed reflection spectrum and multiple machine learning algorithms" 
% By Ying Li, Qinglai Yu, Ming Xie*, Zhenduo Zhang, Zhanjun Ma, and Kai Cao.
% Consult the *Corresponding author: "mingxie@dlmu.edu.cn" in case you have any question with the codes

%% Random forest algorithm for oil type classification
%% Both single tree classification and bagged RF classification are included

% You WILL need to run data preprocessing codes and generate the "database" variable before using this code.

% shuffle dataset
r=randperm(size(database,1)); 
shuffled_data=database(r,:);

% generate training and testing data, as well as their labels
train_data=shuffled_data(1:ceil(0.8*size(database,1)),1:(size(database,2)-1));
train_label=shuffled_data(1:ceil(0.8*size(database,1)),size(database,2));
test_data=shuffled_data((ceil(0.8*size(database,1))+1):size(database,1),1:(size(database,2)-1));
test_label=shuffled_data((ceil(0.8*size(database,1))+1):size(database,1),size(database,2));

% single tree classification
% model training
mdl_ctree = ClassificationTree.fit(train_data,train_label);
% model prediction
ypred = predict(mdl_ctree,test_data);
% generate confusion matrix
Confmat_ctree = confusionmat(test_label,ypred);
% results visualization
figure,
heatmap(Confmat_ctree);
title('Confusion Matrix: Single Classification Tree')

%bagged RF classification
% model training
NumTree=200;
mdl_bagged = fitensemble(train_data,train_label,'bag',NumTree,'tree','type','Classification');
% model prediction
ypred = predict(mdl_bagged,test_data);
% generate confusion matrix
Confmat_bag = confusionmat(test_label,ypred);
% results visualization
figure,
heatmap(Confmat_bag);
title('Confusion Matrix: Ensemble of Bagged Classification Trees')