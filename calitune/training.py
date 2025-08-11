import torch
import torch.nn.functional as F
from tqdm import tqdm

from .utils import move_dict_to_device


def combined_train(model, train_loader, optimizer, criterion, device, item_popularity_tensor, logger, alpha=0.5):
    model.train()
    torch.cuda.empty_cache()

    total_loss = 0
    for sample, score_target, pop_target in tqdm(train_loader, desc="Training...", total=len(train_loader)):
        torch.cuda.empty_cache()
        # move to GPU
        sample = move_dict_to_device(sample, device)
        score_target = score_target.to(torch.float32).to(device)
        pop_target = pop_target.to(device)

        user_id = sample['id']
        candidates = sample['candidates']
        optimizer.zero_grad()

        # Get model predictions
        output = model(user_id, candidates)

        """Score Loss"""
        if alpha < 1.0:
            score_prediction = F.log_softmax(output, dim=1)
            score_target = F.softmax(score_target, dim=1)

            score_loss = criterion(score_prediction, score_target)
        else:
            # Score loss scaled to 0, skip computation
            score_loss = torch.tensor(0.0, device=device)

        """Popularity Loss"""
        candidate_popularity = item_popularity_tensor[candidates]

        # batch_size, num_candidates, num_groups = candidate_popularity.shape

        # Create masks for all items and groups for each user
        candidate_popularity = candidate_popularity.permute(0, 2, 1)  # [batch_size, num_groups, num_candidates]

        if alpha > 0.0:
            # Normalize the score predictions within [0, 1]
            restricted_predictions = output - output.min(1, keepdim=True)[0]
            restricted_predictions = restricted_predictions / restricted_predictions.max(1, keepdim=True)[0]
            restricted_predictions = torch.unsqueeze(restricted_predictions, 1)
            pop_score_weights = (restricted_predictions * candidate_popularity).sum(dim=2)
            pop_score_weights = pop_score_weights / torch.sum(pop_score_weights, dim=1, keepdim=True)
            popularity_loss = criterion((pop_score_weights + 1e-10).log(), pop_target + 1e-10)
        else:
            # Pop loss scaled to 0, skip computation
            popularity_loss = torch.tensor(0.0, device=device)

        combined_loss = (1 - alpha) * score_loss + alpha * popularity_loss

        total_loss += combined_loss.abs().item()

        combined_loss.backward()
        optimizer.step()

    # logger.info(f'Total Train loss: {total_loss}')

def train_model(config, model, train_loader, optimizer, criterion, device, item_popularity_tensor, logger):
    mode = config['finetuning_mode']
    if mode == 'rec':
        combined_train(model, train_loader, optimizer, criterion, device, item_popularity_tensor, logger, alpha=0.0)
    elif mode == 'pop':
        combined_train(model, train_loader, optimizer, criterion, device, item_popularity_tensor, logger, alpha=1.0)
    elif mode == 'combined':
        alpha = config.get('combined_mode_alpha', 0.5)
        combined_train(model, train_loader, optimizer, criterion, device, item_popularity_tensor, logger, alpha=alpha)
