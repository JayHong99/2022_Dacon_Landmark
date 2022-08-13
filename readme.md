## DACON Landmark Classification Task
https://dacon.io/competitions/official/235957/overview/description

## Setting Environment

Linux => Install poetry && pyenv for Setting Python Environment
```
 git clone https://github.com/pyenv/pyenv.git ~/.pyenv
 sed -Ei -e '/^([^#]|$)/ {a \
export PYENV_ROOT="$HOME/.pyenv"
a \
export PATH="$PYENV_ROOT/bin:$PATH"
a \
' -e ':a' -e '$!{n;ba};}' ~/.bash_profile
echo 'eval "$(pyenv init --path)"' >> ~/.bash_profile

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
echo 'eval "$(pyenv init --path)"' >> ~/.profile

echo 'eval "$(pyenv init -)"' >> ~/.bashrc

poetry install
poetry shell

source .venv/bin/activate
 ```



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