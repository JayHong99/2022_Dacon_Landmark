import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import pandas as pd
import yaml
from pathlib import Path


from .set_seed import set_seed

with open('configure.yaml') as config : 
    variables = yaml.safe_load(config)

for key, value in variables.items():
    if key.endswith('_path') : 
        globals()[key] = Path(value)
        globals()[key].mkdir(exist_ok = True)
    else : 
        globals()[key] = value

set_seed(random_seed) # SEED 고정 

class Trainer : 
    """
    Module for training
    """
    def __init__(self, model, dataloaders) -> None : 
        """
        Initialize Trainer
        """
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), learning_rate)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, lr_milestones, gamma)
        self.device = device
        self.early_stop = 0
        self.early_stop_criterion = earlystop_criterion
        self.early_stop_path = model_save_path.joinpath('Model_weight').with_suffix('.pth')

    def run_batch(self, phase, X, label) : 
        """
        Train or Eval Batch Data
        """
        with torch.set_grad_enabled(phase == 'train') : 
            X = X.type(torch.FloatTensor).to(self.device, non_blocking = True)
            label = label.to(self.device , non_blocking= True)
            output = self.model(X)
            loss = self.criterion(output, label)
            if phase == 'train' : 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return output, loss
            
    def run_epoch(self, phase) : 
        """
        Run Epoch
        """
        if phase == 'train' : 
            self.model.train()
        else : 
            self.model.eval()
        
        self.epoch_loss = 0
        self.epoch_outputs = []
        self.epoch_labels = []
        with torch.set_grad_enabled(phase == 'train') : 
            for X, label in self.dataloader : 
                output, loss = self.run_batch( phase, X, label)

                if phase == 'train' : 
                    self.scheduler.step()
                self.epoch_loss += loss.item() 
                self.epoch_outputs.extend(F.softmax(output, dim = 1).detach().cpu().numpy())
                self.epoch_labels.extend(label.detach().cpu().numpy())
        self.epoch_outputs = np.array(self.epoch_outputs)
        self.epoch_labels = np.array(self.epoch_labels)
        self.epoch_loss = self.epoch_loss / len(self.dataloader.dataset)

    def calculate_scores(self) : 
        """
        Calculate Score
        """
        hard_prediction = np.argmax(self.epoch_outputs, axis = 1)
        self.epoch_acc = sum(hard_prediction == self.epoch_labels)/ len(self.epoch_outputs)

    def process(self, phase) : 
        """
        Process
        """
        self.dataloader = self.dataloaders[phase]
        self.run_epoch(phase)
        self.calculate_scores()
        print(f'{phase} LOSS : [{str(round(self.epoch_loss, 5)).ljust(7, "0")}]    ACC : [{str(round(self.epoch_acc, 5)).ljust(7, "0")}]')
        if phase == 'valid' : 
            if self.best_val_acc > self.epoch_acc : 
                self.early_stop += 1
                print(f'EARLY STOP : {self.early_stop}')
            else : 
                self.best_val_acc = self.epoch_acc
                self.early_stop = 0
                torch.save(self.model.state_dict(), self.early_stop_path)


    def train(self, num_epochs, validation = True) : 
        """
        Train Model
        """
        self.best_val_acc = 0
        for num_epoch in range(num_epochs) : 
            print(f'\nEPOCH : {num_epoch}')
            self.process('train')
            if validation : 
                self.process('valid')
                if self.early_stop == self.early_stop_criterion : 
                    print("EARLY STOPPED")
                    break
        self.num_epochs = num_epoch - self.early_stop
    
    
    def test(self) : 
        self.dataloader = self.dataloaders['test']
        self.model.load_state_dict(torch.load(self.early_stop_path))
        self.run_epoch('test')
        pred = self.epoch_outputs.argmax(1).astype(np.uint8)
        print("NUM PREDICTED : ", np.bincount(pred))
        output_name = datetime.now().strftime('%Y_%m_%d_%H_%M_output.csv')
        output_path = submission_path.joinpath(output_name)
        print('Model Prediction Saved at : ', output_name)
        pd.DataFrame([[str(x).rjust(3,'0')+'.PNG' for x in range(1, 200)], pred]
                        ).T.rename(columns = {0: 'file_name', 1:'label'}
                        ).to_csv(output_path,index=False)