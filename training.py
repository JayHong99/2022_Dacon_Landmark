
from src.dataloader import CustomDataLoader
from src.resnet50 import ResNet50
from src.trainer import Trainer



def training() : 
    dataloader = CustomDataLoader()
    dataloaders = dataloader.dataloaders()
    model = ResNet50(10)
    trainer = Trainer(model, dataloaders)
    trainer.train(1000)


    print("TOTAL TRAINING INITIATED")
    dataloader.concat_dset()
    new_dataloaders = {'train' : dataloader.single_dataloader(dataloader.train_dset , True)}
    trainer.model = model 
    trainer.dataloaders = new_dataloaders
    trainer.train(1000, validation = False)


if __name__ == "__main__": 
    training()