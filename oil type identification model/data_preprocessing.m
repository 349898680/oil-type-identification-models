% Implements the framework presented in
% "Identifying oil spill types based on remotely sensed reflection spectrum and multiple machine learning algorithms" 
% By Ying Li, Qinglai Yu, Ming Xie*, Zhenduo Zhang, Zhanjun Ma, and Kai Cao.

%% Data preprocessing on the demo data
%% Please consult the corresponding author for the full collection of data.
% The demo data includes the spectrum of four types of oil obtained under three types of thickness and two types of wind condition.
% The six digits in filenames indicate the condition of the data
% The first two digits indicate the oil type: "01" represents crude oil, "02" represents diesel, "03" represents lubricant, "04" represents heavy diesel.
% The two digits in the middle indicate the oil thickness: "01" represents 0.003069 mm, "02" represents 0.05115 mm, "03" represents 1.944 mm.
% The last two digits indicate the wind condition: "01" represents with wind, "02" represents no wind.

% The data preprocessing codes are produced under the condition that none of the variable has more than nine categories.
% If you have more than nine categories in one of the conditions, the way to generate filenames needs to be changed

clc; clear; close all;

database=[]; % Initialize database

OilType_ID=4; % Number of oil type
OilThickness_ID=3; % Number of oil thickness
WindCondition_ID=2; % Number of wind condition

for i=1:OilType_ID
  for j=1:OilThickness_ID
    for k=1:WindCondition_ID
      filename=strcat('./demo_data/0',num2str(i),'0',num2str(j),'0',num2str(k),'.mat');
      newdata=cell2mat(struct2cell(load(filename))); % Load new data
      a=ones(190,1)*i;  % Generate data labels
      newdata=cat(2,newdata,a); % Add data labels for supervised learning
      database=cat(1,database,newdata); % Concatenate final dataset
      end
  end
end