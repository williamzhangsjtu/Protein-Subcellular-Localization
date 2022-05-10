import torch
from torchvision.transforms import *
import numpy as np
import h5py 
import yaml
import torch
import sys
from loguru import logger
from sklearn.model_selection import train_test_split

def split_dataset(input_h5, debug=False, seed=0):
    with h5py.File(input_h5, 'r') as input_data:
        samples = np.unique(input_data['Sample'])
    if debug:
        samples = samples[:len(samples) // 10]
    train, rest = train_test_split(samples, train_size=0.7, random_state=seed)
    dev, test = train_test_split(rest, test_size=0.5, random_state=seed)
    return train, dev, test


def parse_config(config_file):
    with open(config_file) as con_read:
        yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
    return yaml_config

def genlogger(file):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    if file:
        logger.add(file, enqueue=True, format=log_format)
    return logger

TrainTransform = Compose([
    ToPILImage(),
    RandomResizedCrop(224),
    ColorJitter(0.4, 0.4, 0.4, 0.4),
    RandomGrayscale(p=0.2),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010))])


EvalTransform = Compose([
    ToPILImage(),
    Resize(224),
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010))])