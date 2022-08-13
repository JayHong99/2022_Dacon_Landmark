## DACON Landmark Classification Task
https://dacon.io/competitions/official/235957/overview/description

## Description

- Model : ResNet50
- 1st step : Stratified Split Data => train : valid 0.8 : 0.2
- 2nd step : Train Model for find best epoch
- 3rd step : Concat Train-Valid Data into Train Data
- 4th step : Train Model with Whole Data
- 5th step : Inference


### 1st ~ 4th step
```python3
python training.py
```

### 5th step
```python3
python inference.py
```

*caution*
check "configure.yaml" for hyperparameters