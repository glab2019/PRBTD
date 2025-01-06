# Source code for paper entitled "Can We Enhance the Quality of Mobile Crowdsensing Data Without Ground Truth?"
## Requirments
System: Ubuntu 16.04 LTS with 1080Ti (GPU) 

Python: 3.6.3 +

Pytorch version: 1.1.0
## Execution

Addition:
The setting of the following .py files are written in the corresponding codes, please review and modify as necessary before running them.

1. Download and place the .h5 file of TaxiBJ in 'Predict/data'.
   
2. Get to 'Predict/scripts/model' for training a prediction model
```
python train.py
```
The model file and the prediction results are in 'Predict/scripts/model/bj_taxi_result'.
  
3. Get to 'CreateNoise' for adding noise in the dataset
``` 
python add_noise.py
```
The output are 'data.csv' and 'noise.scv' in 'Result/origin/#/#/simulate_data' where '#' is the directory named by the specifical setting.

The other .py files in 'CreateNoise' are to simulate the cases metioned in our paper, run them if necessary.

4. Run the method file in 'Method'
```
python PRBTD.py
```
or other .py for PRBTD or baseline methods, the results are written in 'Result/origin/#/#/result.csv'.

## Note
This paper is under reviewed. The copyright of the code is not disclosed.
