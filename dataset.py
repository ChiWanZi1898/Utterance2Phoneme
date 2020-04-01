import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch


class UtteranceDataset(Dataset):
    def __init__(self, data_path, label_path=None, test=False):
        self.data = np.load(data_path, allow_pickle=True)
        if not test:
            self.label = np.load(label_path, allow_pickle=True)

        self.test = test

    def __getitem__(self, item):
        if not self.test:
            return self.data[item], self.label[item] + 1
        else:
            return self.data[item]

    def __len__(self):
        return len(self.data)


def collate(batch):
    batch_data = [torch.from_numpy(b[0]) for b in batch]
    batch_label = [torch.from_numpy(b[1]) for b in batch]
    batch_input_len = torch.LongTensor([len(b[0]) for b in batch])
    batch_label_len = torch.LongTensor([len(b[1]) for b in batch])
    batch_data = pad_sequence(batch_data, batch_first=False)
    batch_label = pad_sequence(batch_label,  batch_first=True)
    return batch_data, batch_label, batch_input_len, batch_label_len

def test_collate(batch):
    batch_data = [torch.from_numpy(b) for b in batch]
    batch_input_len = torch.LongTensor([len(b) for b in batch])
    batch_data = pad_sequence(batch_data, batch_first=False)
    return batch_data, batch_input_len


def create_dataloader(data_path, label_path, batch_size, shuffle, test=False, pin_memory=True):
    if not test:
        dataset = UtteranceDataset(data_path, label_path)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                            pin_memory=pin_memory)
    else:
        dataset = UtteranceDataset(data_path, label_path, test=True)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=test_collate,
                                pin_memory=pin_memory)
    return dataloader
