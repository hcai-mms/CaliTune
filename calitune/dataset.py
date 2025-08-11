import torch
from numpy import ndarray
from torch.utils.data import Dataset


class CalituneDataset(Dataset):
    def __init__(self, candidates: ndarray, score_targets: torch.Tensor, popularity_targets: ndarray):
        self.candidates = candidates
        self.targets = score_targets
        self.popularity_targets = popularity_targets

    def __len__(self):
        return len(self.candidates)

    def __getitem__(self, idx:int) -> tuple[dict, ndarray]:
        """Get item....

        Args:
            idx (int): index of sample to return

        Returns:
            tuple[dict, ndarray]: A tuple containing:
                -dict: dictionary of user IDs(int) and its candidates(torch.Tensor)
                -ndarray: target vector as ndarray

        """
        candidates = self.candidates[idx]
        candidate_tensor = torch.tensor(candidates, dtype=torch.int)
        sample = {'id': idx, 'candidates': candidate_tensor}
        target = self.targets[idx]
        popularity_target = self.popularity_targets[idx]
        return sample, target, popularity_target