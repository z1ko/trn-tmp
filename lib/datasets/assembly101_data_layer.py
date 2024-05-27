import torch
import pickle
import einops
import os
import tqdm
import numpy
import random

DEC_STEPS = 15
ENC_STEPS = 200 #1000

class Assembly101Dataset(torch.utils.data.Dataset):
    def __init__(self, mode, config, target):

        self.dec_steps = DEC_STEPS
        self.enc_steps = ENC_STEPS
        self.target = target

        # load classes weights
        with open('/home/fziche/develop/TRN.pytorch/data/Assembly101/fine-labels-weights.pkl', 'rb') as f:
            self.weights = [torch.tensor(x, dtype=torch.float32).cuda() 
                            for x in pickle.load(f)]

        self.items = []
        split_path = os.path.join('/home/fziche/develop/TRN.pytorch/data/Assembly101/processed', mode)
        for item in tqdm.tqdm(os.listdir(split_path)):
            item_path = os.path.join(split_path, item)
            
            #self.items.append((item_path, None))

            if mode == 'train':
                frames = int(item.split('-')[0])

                seed = random.randint(0, frames - self.dec_steps)
                for start, end in zip(
                    range(seed, frames - self.dec_steps, self.enc_steps),
                    range(seed + self.enc_steps, frames - self.dec_steps, self.enc_steps)):

                    self.items.append((item_path, (start, end)))
            else:
                self.items.append((item_path, None))

    def get_dec_target(self, target_vector):
        target_matrix = numpy.zeros((self.enc_steps, self.dec_steps))
        for i in range(self.enc_steps):
            for j in range(self.dec_steps):
                target_matrix[i,j] = target_vector[i+j]
        return target_matrix

    def __getitem__(self, idx):
        path, clip = self.items[idx]
        with open(path, 'rb') as f:
            sample = pickle.load(f)

        action_map = {
            'verb': 1,
            'noun': 2
        }

        labels = torch.tensor(sample['fine-labels']).long()
        labels = labels[..., action_map[self.target]]

        embeddings = torch.tensor(sample['embeddings'], dtype=torch.float32)
        poses = torch.tensor(sample['poses'],dtype=torch.float32)
        poses = einops.rearrange(poses, 'T H J F -> T (H J) F')

        if clip is not None:
            start, end = clip

            enc_labels = labels[start:end, ...] 
            dec_labels = labels[start:end + self.dec_steps, ...]
            dec_labels = self.get_dec_target(dec_labels)

            embeddings = embeddings[start:end, ...]
            poses = poses[start:end, ...]

        else:
            enc_labels = labels
            dec_labels = torch.tensor((1,))

        return embeddings, poses, enc_labels, dec_labels

    def __len__(self):
        return len(self.items)
