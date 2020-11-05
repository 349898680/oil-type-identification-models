# Identifying oil spill types based on remotely sensed reflection spectrum and multiple machine learning algorithms
 
This demo includes demo data and MATLAB codes for the paper "Identifying oil spill types based on remotely sensed reflection spectrum and multiple machine learning algorithms"
by Ying Li, Qinglai Yu, Ming Xie*, Zhenduo Zhang, Zhanjun Ma, and Kai Cao.

*Corresponding author: mingxie@dlmu.edu.cn.

Warning: the code was tested using MATLAB R2019a and it might be incompatible with older versions.  

The demo data is included in the "demo_data" foulder. Please consult the corresponding author for the full collection of data.
The demo data includes the spectrum of four types of oil obtained under three types of thickness and two types of wind condition.
The six digits in filenames indicate the condition of the data
The first two digits indicate the oil type: "01" represents crude oil, "02" represents diesel, "03" represents lubricant, "04" represents heavy diesel.
The two digits in the middle indicate the oil thickness: "01" represents 0.003069 mm, "02" represents 0.05115 mm, "03" represents 1.944 mm.
The last two digits indicate the wind condition: "01" represents with wind, "02" represents no wind.


Note: This oil type identification algorithm mainly consists of two parts:
       - A data preprocessing algorithm 
       - Three commonly-used machine learning models

To use the codes:
    - Run the data preprocessing algorithm first and generate "database" variable,
       It is a 2D cell all the labelled training and testing data.
    - Choose to run one of the machine learning models.
    - You may need to clear all the variables and reload the "database" variable when you finish one of the algorithm and want to try another.
