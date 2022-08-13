from PIL import Image
import numpy as np
from joblib import Parallel, delayed

    


def calculate_norm(data_paths : list) -> tuple : 

    images = Parallel()(delayed(load_images)(data_path) for data_path in data_paths)
    images = np.array(images)
    image_mean = np.array([np.mean(images[:,:,:,0]/255), np.mean(images[:,:,:,1]/255), np.mean(images[:,:,:,2]/255)]) + 1e-5
    image_std  = np.array([np.std(images[:,:,:,0]/255) , np.std(images[:,:,:,1]/255) , np.std(images[:,:,:,2]/255) ]) + 1e-5
    print(f"Image Mean : {image_mean} | Image std : {image_std}")
    return image_mean, image_std

def load_images(data_path) : 
    """
    이미지들 다 쌓아서, RGB로 Image mean, std 구함
    """
    return np.array(Image.open(data_path))