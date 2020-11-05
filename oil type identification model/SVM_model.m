% Implements the framework presented in
% "Identifying oil spill types based on remotely sensed reflection spectrum and multiple machine learning algorithms" 
% By Ying Li, Qinglai Yu, Ming Xie*, Zhenduo Zhang, Zhanjun Ma, and Kai Cao.
% Consult the *Corresponding author: "mingxie@dlmu.edu.cn" in case you have any question with the codes

%% Support vector machine algorithm for oil type classification
%% Three SVM classifiers are trained to classify four types of oil
%% Change the number of SVM classifiers that need to be trained if you have different number of categories.

% You WILL need to run data preprocessing codes and generate the "database" variable before using this code.

% shuffle dataset
r=randperm(size(database,1)); 
shuffled_data=database(r,:);

% generate overall training and testing data, as well as their labels
train_data=shuffled_data(1:ceil(0.8*size(database,1)),1:(size(database,2)-1));
train_label=shuffled_data(1:ceil(0.8*size(database,1)),size(database,2));
test_data=shuffled_data((ceil(0.8*size(database,1))+1):size(database,1),1:(size(database,2)-1));
test_label=shuffled_data((ceil(0.8*size(database,1))+1):size(database,1),size(database,2));

% preparing three sets of traning data for the three classifiers
train_label_s1=[];
train_data_s2=[];
train_label_s2=[];
train_data_s3=[];
train_label_s3=[];
s2=1;
s3=1;
for m=1:ceil(0.8*size(database,1))
    if train_label(m,1)==2 || train_label(m,1)==3
       train_label_s1(m,1)=1;
       train_data_s2=cat(1,train_data_s2,train_data(m,:));
       train_label_s2(s2,1)=2;
       train_data_s3=cat(1,train_data_s3,train_data(m,:));
       train_label_s3(s3,1)=train_label(m,1);
       s2=s2+1;
       s3=s3+1;
    elseif train_label(m,1)==1
       train_label_s1(m,1)=1;
       train_data_s2=cat(1,train_data_s2,train_data(m,:));
       train_label_s2(s2,1)=1;
       s2=s2+1;
    else
       train_label_s1(m,1)=4;       
    end
end

% train the three classifiers
mdl_svm1 = fitcsvm(train_data,train_label_s1,'KernelFunction','RBF','KernelScale','auto');
mdl_svm2 = fitcsvm(train_data_s2,train_label_s2,'KernelFunction','RBF','KernelScale','auto');
mdl_svm3 = fitcsvm(train_data_s3,train_label_s3,'KernelFunction','RBF','KernelScale','auto');

% model prediction
ypred=[];
for n=1:912
    if predict(mdl_svm1,test_data(n,:))==4
       ypred(n,1) = 4;
    elseif  predict(mdl_svm2,test_data(n,:))==1
       ypred(n,1) = 1;
    elseif  predict(mdl_svm3,test_data(n,:))==2
       ypred(n,1) = 2;
    else
       ypred(n,1) = 3;
    end
end

% generate confusion matrix
Confmat_svm = confusionmat(test_label,ypred);

% results visualization
figure,
heatmap(Confmat_svm);
title('Confusion Matrix: Support Vector Machine')