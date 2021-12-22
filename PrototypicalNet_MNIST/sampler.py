import numpy as np
import torch

class PrototypeSampler(torch.utils.data.Sampler):

    def __init__(self, target, n_support, n_query, n_way, iterations) -> None:
        self.target = target
        self.classes = torch.unique(target)
        self.num_samples = n_support + n_query
        self.num_cls_per_it = n_way
        self.iterations = iterations

    def __iter__(self):

        for it in range(self.iterations):
            batch = [] # batch_size = Nc * (Ns + Nq)
            K = np.random.choice(self.classes, self.num_cls_per_it, replace=False)
            for i, cls in enumerate(K):
                idxs = self.target.eq(cls).nonzero().squeeze()
                sample_idxs = torch.randperm(len(idxs))[:self.num_samples]
                batch.append(idxs[sample_idxs])
            batch = torch.cat(batch)
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        return self.iterations    