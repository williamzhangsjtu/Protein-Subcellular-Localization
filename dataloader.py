from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import h5py
import numpy as np
import torch

class ProteinDataset(Dataset):
    def __init__(self, input_h5, samples, transform):
        self.input_h5 = input_h5
        self.samples = samples
        self.indices = []
        with h5py.File(input_h5, 'r') as input_data:
            for sample in samples:
                self.indices.extend(
                    np.where(input_data['Sample'][()] == sample)[0])
        self.transform = transform
        
    def __getitem__(self, i):
        idx = self.indices[i]
        with h5py.File(self.input_h5, 'r') as input_data:
            data = input_data['Image'][idx].astype(np.uint8)
            sample = input_data['Sample'][idx].decode()
            target = input_data['target'][idx]
            instance = input_data['Instance'][idx].decode()
        return self.transform(data),\
            torch.tensor(target, dtype=torch.float), sample, instance
        
    def __len__(self):
        return len(self.indices)

class MultiInstanceSampler():
    """
    batch sampler
    """
    def __init__(self, input_h5, samples, shuffle=True, seed=0, **kwargs):
        self.random_state = np.random.RandomState(seed)
        self.n_instances = kwargs.get('n_instances', None)
        self.batch_size = kwargs.get('batch_size', 4)

        self.samples = samples
        self.sample_indices = list(range(len(samples)))
        if shuffle:
            self.random_state.shuffle(self.sample_indices)
        self.ptr_sample = 0

        self.indices_per_sample = [[] for _ in range(len(samples))]
        self.ptr_per_sample = [0] * len(samples)
        with h5py.File(input_h5, 'r') as input_data:
            for idx, sample in enumerate(samples):
                self.indices_per_sample[idx].extend(
                    np.where(input_data['Sample'][()] == sample)[0])
                if shuffle:
                    self.random_state.shuffle(self.indices_per_sample[idx])
        self.shuffle = shuffle
    def _get_data(self):
        sample_idx = self.sample_indices[self.ptr_sample]
        self.ptr_sample = (self.ptr_sample + 1) % len(self.sample_indices)
        if not self.ptr_sample and self.shuffle:
            self.random_state.shuffle(self.sample_indices)
        indices = []
        if self.n_instances is None:
            self.random_state.shuffle(self.indices_per_sample[sample_idx])
            return self.indices_per_sample[sample_idx]
        ptr = self.ptr_per_sample[sample_idx]
        for _ in range(self.n_instances):
            indices.append(self.indices_per_sample[sample_idx][ptr])
            ptr = (ptr + 1) % len(self.indices_per_sample[sample_idx])
            if not ptr and self.shuffle:
                self.random_state.shuffle(self.indices_per_sample[sample_idx])
        self.ptr_per_sample[sample_idx] = ptr
            
        return indices
        
    def __iter__(self):
        while True:
            batch_data = []
            for _ in range(self.batch_size):
                batch_data.append(self._get_data())
            yield batch_data

class MultiInstanceDataset(Dataset):
    def __init__(self, input_h5, transform):
        self.input_h5 = input_h5
        self.transform = transform

    def __getitem__(self, indices):
        """
        indices: list of index
        """
        data, instances = [], []
        with h5py.File(self.input_h5, 'r') as input_data:
            for idx in indices:
                raw_data = input_data['Image'][idx].astype(np.uint8)
                data.append(self.transform(raw_data))
                instance = input_data['Instance'][idx].decode()
                instances.append(instance)
                sample = input_data['Sample'][idx].decode()
                target = input_data['target'][idx]
        data = torch.stack(data, dim=0) # N x 3 x H x W
        return data, torch.tensor(target, dtype=torch.float),\
            sample, instances

class MultiInstanceEvalDataset(Dataset):
    def __init__(self, input_h5, samples, transform, n_instances=None, seed=0, n_repeat=1):
        self.input_h5 = input_h5
        self.transform = transform
        self.samples = samples
        self.instance_bag = []
        self.sample_bag = []
        self.random_state = np.random.RandomState(seed)
        with h5py.File(self.input_h5, 'r') as input_data:
            for _ in range(n_repeat):
                for sample in samples:
                    indices = np.where(input_data['Sample'][()] == sample)[0]
                    self.random_state.shuffle(indices)
                    if n_instances is None: # all instances at one time
                        self.instance_bag.append(indices)
                        self.sample_bag.append(sample)
                        continue
                    n = (len(indices) // n_instances + 1) * n_instances
                    bag = []
                    for ptr in range(n):
                        ptr = ptr % len(indices)
                        bag.append(indices[ptr])
                        if not len(bag) % n_instances:
                            self.instance_bag.append(bag)
                            self.sample_bag.append(sample)
                            bag = []
                    if bag:
                        self.instance_bag.append(bag)
                        self.sample_bag.append(sample)
                        bag = []
                
    
    def __getitem__(self, i):
        indices = self.instance_bag[i]
        sample = self.sample_bag[i].decode()
        data, names = [], []
        with h5py.File(self.input_h5, 'r') as input_data:
            for idx in indices:
                raw_data = input_data['Image'][idx].astype(np.uint8)
                data.append(self.transform(raw_data))
                name = input_data['Instance'][idx].decode()
                names.append(name)
                target = input_data['target'][idx]
        data = torch.stack(data, dim=0) # N x 3 x H x W
        return data,\
            torch.tensor(target, dtype=torch.float), sample, names
    
    def __len__(self):
        return len(self.sample_bag)

def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    data, targets, samples, names = zip(*batch)
    data = pad_sequence(data, batch_first=True)
    return data, torch.stack(targets, dim=0), samples, names

def create_multi_instance_dataloader(
    input_h5: str, 
    samples: list,
    transform,
    seed: int = 0,
    shuffle: bool = True,
    **kwargs
):
    sampler_args = kwargs.get('sampler_args', 
        {'n_instances': 4, 'batch_size': 8, 'n_repeat': 1})
    dataloader_args = kwargs.get('dataloader_args', {'num_workers': 8})
    if shuffle:
        dataset = MultiInstanceDataset(input_h5, transform)
        sampler = MultiInstanceSampler(input_h5, samples,\
            seed=seed, shuffle=shuffle, **sampler_args)
        return DataLoader(dataset,
            batch_sampler=sampler, collate_fn=collate_fn, **dataloader_args)
    else:
        dataloader_args['batch_size'] = sampler_args['batch_size']
        dataset = MultiInstanceEvalDataset(
            input_h5=input_h5,
            samples=samples,
            transform=transform,
            n_instances=sampler_args['n_instances'],
            seed=seed,
            n_repeat=sampler_args.get('n_repeat', 1)
        )
        return DataLoader(dataset,
            shuffle=False, collate_fn=collate_fn, **dataloader_args)




def create_dataloader(input_h5, samples, transform, **kwargs):
    kwargs.setdefault('batch_size', 32)
    kwargs.setdefault('shuffle', True)
    kwargs.setdefault('num_workers', 8)
    dataset = ProteinDataset(input_h5, samples, transform)
    return DataLoader(dataset, **kwargs)