
from src.dataloader import CustomDataLoader
from src.resnet50 import ResNet50
from src.trainer import Trainer


def inference() : 
    dataloaders = CustomDataLoader()
    dataloaders = {'test' : dataloaders.single_dataloader(dataloaders.test_dset , False)}
    model = ResNet50(10)
    trainer = Trainer(model, dataloaders)
    trainer.test()



if __name__ == "__main__":
    inference()
