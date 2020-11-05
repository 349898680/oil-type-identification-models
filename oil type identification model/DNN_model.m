% Implements the framework presented in
% "Identifying oil spill types based on remotely sensed reflection spectrum and multiple machine learning algorithms" 
% By Ying Li, Qinglai Yu, Ming Xie*, Zhenduo Zhang, Zhanjun Ma, and Kai Cao.
% Consult the *Corresponding author: "mingxie@dlmu.edu.cn" in case you have any question with the codes

%% Deep neural network algorithm for oil type classification

% You WILL need to run data preprocessing codes and generate the "database" variable before using this code.

% shuffle dataset
r=randperm(size(database,1)); 
shuffled_data=database(r,:);

% generate training and testing data, as well as their labels
train_data=shuffled_data(1:ceil(0.8*size(database,1)),1:(size(database,2)-1));
train_label=shuffled_data(1:ceil(0.8*size(database,1)),size(database,2));
test_data=shuffled_data((ceil(0.8*size(database,1))+1):size(database,1),1:(size(database,2)-1));
test_label=shuffled_data((ceil(0.8*size(database,1))+1):size(database,1),size(database,2));

train_data=train_data';
test_data=test_data';
train_label_dnn=categorical(train_label); % convert to categorical label
inputSize = 1936; % dimension of input, reprensts for 1936 pixel in the dimension of spectrum
numHiddenUnits = 512; % number of neurons in each fully-connected layer
numClasses = 4; % number of classes, reprensents four types of oils

% setup layers
layers = [ ...
    sequenceInputLayer(inputSize)
    fullyConnectedLayer(numClasses)
    fullyConnectedLayer(numClasses)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

maxEpochs = 1000; %maximum number of epochs
miniBatchSize = 27; % batch size

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'Verbose',false, ...
    'Plots','training-progress');

% train model
net=trainNetwork(train_data,train_label_dnn,layers, options);

% model prediction
ypred = classify(net,testdata, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest');

% generate confusion matrix
ypred_num=double(ypred);
Confmat_svm = confusionmat(test_label,ypred_num);

%results visualization
figure,
heatmap(Confmat_svm);
title('Confusion Matrix: Deep Neural Network')
