import torch
import torch.nn as nn


class CaliBPR(nn.Module):
    def __init__(self, init_weights_path: str, interaction_matrix: torch.Tensor, freeze_item_emb=False):
        super(CaliBPR, self).__init__()

        init_weights = torch.load(init_weights_path, map_location=lambda storage, loc: storage)

        self.user_emb = nn.Embedding(
            num_embeddings=init_weights['user_embedding.weight'].size(0),
            embedding_dim=init_weights['user_embedding.weight'].size(1),
            dtype=torch.float32
        )
        self.item_emb = nn.Embedding(
            num_embeddings=init_weights['item_embedding.weight'].size(0),
            embedding_dim=init_weights['item_embedding.weight'].size(1),
            dtype=torch.float32
        )

        self.user_emb.weight.data.copy_(init_weights['user_embedding.weight'])
        self.item_emb.weight.data.copy_(init_weights['item_embedding.weight'])

        if freeze_item_emb:
            self.item_emb.weight.requires_grad = False

    def full_predict(self, users, batch_size=100):
        num_users = len(users)

        scores_list = []  # Store batches instead of allocating full tensor

        with torch.no_grad():
            for i in range(0, num_users, batch_size):
                start_idx = i
                end_idx = min(i + batch_size, num_users)

                user_embeddings = self.user_emb.weight[users[start_idx:end_idx]]
                item_embeddings = self.item_emb.weight

                batch_scores = torch.matmul(user_embeddings, item_embeddings.transpose(0, 1))
                # Set padding item scores to -inf
                batch_scores[:, 0] = -torch.inf
                # batch_scores[0, :] = -torch.inf

                scores_list.append(batch_scores.cpu())  # Move batch to CPU to free GPU memory

                # Free memory
                del user_embeddings, batch_scores
                torch.cuda.empty_cache()

        # Concatenate only after all batches are processed
        scores = torch.cat(scores_list).cpu()
        return scores

    def forward(self, user, item):
        # Get user embeddings of shape (batch_size, emb_dim)
        user_embeddings = self.user_emb(user)

        # Get item embeddings of shape (batch_size, num_candidates, emb_dim)
        item_embeddings = self.item_emb(item)

        # Multiply: broadcasting user embeddings to (batch_size, 1, emb_dim) and then element-wise multiply.
        # Sum over the embedding dimension to get the dot product for each candidate.
        return (user_embeddings.unsqueeze(1) * item_embeddings).sum(dim=-1) # should have shape batch_size x num_candidates


class CaliDMF(nn.Module):
    def __init__(self, init_weights_path: str, interaction_matrix: torch.Tensor, freeze_item_emb=False):
        super(CaliDMF, self).__init__()

        self.interaction_matrix = interaction_matrix
        self.interaction_matrix.requires_grad = False

        # Load the model from the given path
        init_weights = torch.load(init_weights_path, map_location=lambda storage, loc: storage)

        self.user_linear = nn.Linear(
            in_features=init_weights['user_linear.weight'].shape[1],
            out_features=init_weights['user_linear.weight'].shape[0],
            bias=False
        )
        self.user_linear.weight.data.copy_(init_weights['user_linear.weight'])

        self.item_linear = nn.Linear(
            in_features=init_weights['item_linear.weight'].shape[1],
            out_features=init_weights['item_linear.weight'].shape[0],
            bias=False
        )
        self.item_linear.weight.data.copy_(init_weights['item_linear.weight'])

        if freeze_item_emb:
            self.item_linear.weight.requires_grad = False

        # We subtract 2 from the weight count (weights without bias for first linear user/item layers) and divide by 4
        # (weights+bias for each layer) to get the number of hidden layers.
        # It is assumed that the item and user stacks have the same number of layers.
        num_hidden_layers = (len(init_weights) - 2) // 4

        user_stack = []
        item_stack = []
        for i in range(num_hidden_layers):
            # State_dict is indexed like this due to a (disabled) dropout layer before the fully connected layer and a ReLU
            # activation function after the fully connected layer.
            linear_idx = 1 + 3 * i
            user_lin = nn.Linear(
                in_features=init_weights[f'user_fc_layers.mlp_layers.{linear_idx}.weight'].shape[1],
                out_features=init_weights[f'user_fc_layers.mlp_layers.{linear_idx}.weight'].shape[0],
                bias=True
            )
            user_lin.weight.data.copy_(init_weights[f'user_fc_layers.mlp_layers.{linear_idx}.weight'])
            user_lin.bias.data.copy_(init_weights[f'user_fc_layers.mlp_layers.{linear_idx}.bias'])
            user_stack.append(user_lin)
            user_stack.append(nn.ReLU())

            item_lin = nn.Linear(
                in_features=init_weights[f'item_fc_layers.mlp_layers.{linear_idx}.weight'].shape[1],
                out_features=init_weights[f'item_fc_layers.mlp_layers.{linear_idx}.weight'].shape[0],
                bias=True
            )
            item_lin.weight.data.copy_(init_weights[f'item_fc_layers.mlp_layers.{linear_idx}.weight'])
            item_lin.bias.data.copy_(init_weights[f'item_fc_layers.mlp_layers.{linear_idx}.bias'])

            if freeze_item_emb:
                item_lin.weight.requires_grad = False
                item_lin.bias.requires_grad = False

            item_stack.append(item_lin)
            item_stack.append(nn.ReLU())

        self.user_stack = nn.Sequential(*user_stack)
        self.item_stack = nn.Sequential(*item_stack)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user, item):
        user_embeddings = self.user_linear(self.interaction_matrix[user])
        user_embeddings = self.user_stack(user_embeddings)

        item_columns = self.interaction_matrix[:, item]
        item_columns = item_columns.permute(1, 2, 0) # (batch_size, k_candidates, num_items)

        item_embeddings = self.item_linear(item_columns)
        item_embeddings = self.item_stack(item_embeddings)

        # Multiply: broadcasting user embeddings to (batch_size, 1, emb_dim) and then element-wise multiply.
        # Sum over the embedding dimension to get the dot product for each candidate.
        return (user_embeddings.unsqueeze(1) * item_embeddings).sum(dim=-1) # should have shape batch_size x num_candidates

    def full_predict(self, users, batch_size=100):
        num_users = len(users)

        scores_list = []  # Store batches instead of allocating full tensor

        with torch.no_grad():
            self.interaction_matrix = self.interaction_matrix.to(torch.float32).to(self.user_linear.weight.device)

            candidate_columns = self.interaction_matrix.t().unsqueeze(0)  # shape: (1, num_items, num_users)
            candidate_embeddings = self.item_linear(candidate_columns)      # shape: (1, num_items, out_dim)
            candidate_embeddings = self.item_stack(candidate_embeddings)      # shape: (1, num_items, out_dim)
            candidate_embeddings = candidate_embeddings.squeeze(0)            # shape: (num_items, out_dim)

            scores_list = []
            for i in range(0, num_users, batch_size):
                start_idx = i
                end_idx = min(i + batch_size, num_users)

                user_embeddings = self.user_linear(self.interaction_matrix[users[start_idx:end_idx]])
                user_embeddings = self.user_stack(user_embeddings)

                batch_scores = torch.matmul(user_embeddings, candidate_embeddings.t())
                # Set padding item scores to -inf
                batch_scores[:, 0] = -torch.inf

                scores_list.append(batch_scores.cpu())

                # Free memory
                del user_embeddings, batch_scores
                torch.cuda.empty_cache()

        # Concatenate only after all batches are processed
        scores = torch.cat(scores_list).cpu()
        # Doesn't change order, but Recbole implementation also normalizes the scores through a sigmoid
        scores = self.sigmoid(scores)
        return scores