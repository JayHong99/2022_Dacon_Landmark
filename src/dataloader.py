from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from PIL import Image
from sklearn.model_selection import train_test_split

from .set_seed import set_seed
from .calculate import calculate_norm
from .transforms import train_transforms, test_transforms

with open('configure.yaml') as config : 
    variables = yaml.safe_load(config)

for key, value in variables.items():
    if key.endswith('_path') : 
        globals()[key] = Path(value)
        globals()[key].mkdir(exist_ok = True)
    else : 
        globals()[key] = value

set_seed(random_seed) # SEED 고정 

class CustomDataset : 
    """
    Custom Dataset
    """
    def __init__(self, image_paths, labels, transform = None) : 
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __getitem__(self, index) : 
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        label = self.labels[index]

        if self.transform : 
            image = self.transform(image)
        return image, label        


    def __len__(self) : 
        return len(self.image_paths)


class CustomDataLoader : 
    """
    Custom DataLoader
    """
    def __init__(self) -> None: 
        """
        Initialize Class
        Arguments are loaded from configure.txt
        """
        train_image_path = data_root_path.joinpath(train_image_dir_name)
        test_image_path = data_root_path.joinpath(test_image_dir_name)
        train_label_path = data_root_path.joinpath(train_label_csv_name).with_suffix('.csv')
        self.train_labels = pd.read_csv(train_label_path)['label'].to_numpy()
        train_image_paths = self.load_paths(train_image_path)
        test_image_paths = self.load_paths(test_image_path)

        mean, std = calculate_norm(train_image_paths)
        self.train_transforms = train_transforms(mean, std)
        self.test_transforms = test_transforms(mean, std)

        self.set_dataset(train_image_paths, test_image_paths)


    def load_paths(self, image_path : Path) -> np.array : 
        """
        load PNG Paths in directory
        """
        return np.array([path for path in image_path.glob('*.PNG')])


    def set_dataset(self, train_image_paths : list, test_image_paths : list) -> None : 
        """
        Set train, valid, test dset
        test labels are 1 for inital value
        """
        idx_array = np.arange(len(train_image_paths))
        train_idx, valid_idx = train_test_split(idx_array,
                                                train_size = train_ratio, 
                                                random_state = random_seed,
                                                shuffle = True,
                                                stratify = self.train_labels)

        self.train_dset = CustomDataset(train_image_paths[train_idx], self.train_labels[train_idx],  transform = self.train_transforms)
        self.valid_dset = CustomDataset(train_image_paths[valid_idx], self.train_labels[valid_idx],  transform = self.test_transforms)
        self.test_dset  = CustomDataset(test_image_paths, [1 for x in range(len(test_image_paths))], transform = self.test_transforms)


    def concat_dset(self)  :
        self.train_dset = ConcatDataset([self.train_dset, self.valid_dset])


    def single_dataloader(self, dset : Dataset, shuffle = False) -> DataLoader : 
        """
        Set Single Dataloader
        """
        return DataLoader(dset, batch_size = batch_size, shuffle = shuffle, pin_memory = True)


    def dataloaders(self) -> dict: 
        """
        Set Dataloaders
        """
        return {
                "train" : self.single_dataloader(self.train_dset, True),
                "valid" : self.single_dataloader(self.valid_dset, False),
                "test"  : self.single_dataloader(self.test_dset , False),
                }