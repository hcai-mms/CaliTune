import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .dataset import CalituneDataset
from .evaluation import evaluate
from .logging import setup_logger, log_config
from .models import CaliBPR, CaliDMF
from .training import train_model
from .utils import read_config, get_cuda_device, set_seed, load_embeddings, load_splits, select_candidates, \
    load_popularity_targets, get_targets_for_splits, load_popularity, save_embedding, load_candidate_score_targets


def run_calitune(config, dataset: str):
    logger = setup_logger()
    # Load configuration parameters into variables
    data_path = config['data_path']
    random_seed = config['random_seed']

    # Define Paths from dataset name
    base_weights_path = os.path.join(data_path, dataset, 'base_weights.pth')
    user_embeddings_path = os.path.join(data_path, dataset, f"{dataset}.useremb")
    item_embeddings_path = os.path.join(data_path, dataset, f"{dataset}.itememb")
    user_baseline_path = os.path.join(data_path, dataset, f"{dataset}.userbaseline")
    item_popularity_path = os.path.join(data_path, dataset, f"{dataset}.popularity")

    logger.info(f"Dataset: {dataset}")
    log_config(logger, config)

    device = get_cuda_device(config, logger)

    # Set random seed for reproducibility
    set_seed(random_seed)

    # Import Baseline Item and User Embeddings

    # load popularity information
    item_popularity = load_popularity(item_popularity_path)
    item_popularity = item_popularity.to(device)

    """
    # Create model with baseline embeddings
    model = CaliBPR(
        user_embeddings, item_embeddings,
        user_id_list, item_id_list,
        freeze_item_emb=config['freeze_item_emb']
    )
    """

    # load splits (user IDs with positive item IDs for each split)
    positive_train, positive_valid, positive_test, unique_items, unique_users = load_splits(
        os.path.join(config['splits_folder'], dataset))

    # get y_true matrices
    y_true_train = get_targets_for_splits(positive_train, unique_users, unique_items)
    y_true_valid = get_targets_for_splits(positive_valid, unique_users, unique_items)
    y_true_test = get_targets_for_splits(positive_test, unique_users, unique_items)
    popularity_targets = load_popularity_targets(user_baseline_path)

    if config['model'] == 'CaliBPR':
        model = CaliBPR(
            base_weights_path, y_true_train,
            freeze_item_emb=config['freeze_item_emb']
        )
    elif config['model'] == 'CaliDMF':
        model = CaliDMF(base_weights_path, y_true_train,
                        freeze_item_emb=config['freeze_item_emb'])
    else:
        raise ValueError(f'Unknown Finetuning model: {config["model"]}')

    model.to(device)

    candidates = select_candidates(config, model, unique_users, unique_items, device)
    candidate_y_true = load_candidate_score_targets(y_true_train, candidates)


    train_data = CalituneDataset(candidates, candidate_y_true, popularity_targets)
    train_loader = DataLoader(train_data, batch_size=config['train_batch_size'], shuffle=True)
    learning_rate = config['learning_rate']
    schedule_lr = config['schedule_lr']
    scheduler_step_size = config['lr_scheduler_step_size']
    scheduler_gamma = config['lr_scheduler_gamma']
    # build optimizer
    optimizer = optim.RAdam(model.parameters(), lr=learning_rate)
    # scheduler for decaying LR
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    loss_param = config['loss_param']

    criterion = nn.KLDivLoss(reduction='batchmean')
    if loss_param.casefold() == 'kl':
        criterion = nn.KLDivLoss(reduction='batchmean')
    else:
        logger.error(f"Unknown loss param: {loss_param}")
        raise ValueError(f"Unknown loss param: {loss_param}")


    init_valid_ndcg, init_test_ndcg, init_jsd = evaluate(
        config, True, True, unique_users, device, model,
        y_true_train, y_true_valid, y_true_test,
        popularity_targets, item_popularity, logger
    )

    # Silly idea: fill model with random embeddings
    # model.user_emb.weight.data.copy_(torch.randn(model.user_emb.weight.size()).to(device))
    # model.item_emb.weight.data.copy_(torch.randn(model.item_emb.weight.size()).to(device))

    num_epochs = config['num_epochs']
    early_stopping_patience = config['early_stopping_patience']
    target_metric = config['target_metric']
    model_save_interval = config['model_save_interval']

    last_improvement_epoch = 0
    improvement_counter = 0
    identical_results_counter = 0
    best_ndcg = 0
    best_jsd = 1
    best_tradeoff = 0
    best_weights = model.state_dict()


    valid_ndcg = 0
    jsd = 0
    for epoch in range(num_epochs):
        logger.info(64 * "#")
        logger.info(f"Epoch: {epoch + 1}/{num_epochs}")

        train_model(config, model, train_loader, optimizer, criterion, device, item_popularity, logger)

        # lr schedule
        if schedule_lr:
            scheduler.step()

        if config['regenerate_candidates']:
            candidates = select_candidates(config, model, unique_users, unique_items, device)
            candidate_y_true = load_candidate_score_targets(y_true_train, candidates)
            train_data = CalituneDataset(candidates, candidate_y_true, popularity_targets)
            train_loader = DataLoader(train_data, batch_size=config['train_batch_size'], shuffle=True)

        old_valid_ndcg = valid_ndcg
        old_jsd = jsd
        valid_ndcg, _, jsd = evaluate(config, False, True, unique_users, device,
                                      model,
                                      y_true_train, y_true_valid, y_true_test,
                                      popularity_targets, item_popularity, logger)

        ndcg_ratio = valid_ndcg / init_valid_ndcg
        jsd_ratio = init_jsd / jsd

        # Can be adjusted to influence the relevance of the two metrics
        # 0.5 is equal to the tradeoff value "1" sitting on a straight 45°
        # line in normalized NDCG/JSD space. Values <0.5 put more weight on
        # NDCG (making the line become concave), values >0.5 put more weight
        # on JSD (making the line become convex)
        WEIGHT_NDCG = 0.5

        tradeoff = (ndcg_ratio ** WEIGHT_NDCG) * (jsd_ratio ** (1 - WEIGHT_NDCG))
        logger.info(f'Tradeoff: {tradeoff}')

        if target_metric in ('ndcg', 'any'):
            if valid_ndcg > best_ndcg:
                best_weights = model.state_dict()
                # best_user_emb = model.user_emb.weight.data.cpu().numpy()
                # best_item_emb = model.item_emb.weight.data.cpu().numpy()
                best_ndcg = valid_ndcg
                last_improvement_epoch = epoch
                improvement_counter += 1
                logger.info(f"  New best NDCG: {best_ndcg}")

        if target_metric in ('pop_jsd', 'any'):
            if jsd < best_jsd:
                best_weights = model.state_dict()
                # best_user_emb = model.user_emb.weight.data.cpu().numpy()
                # best_item_emb = model.item_emb.weight.data.cpu().numpy()
                best_jsd = jsd
                last_improvement_epoch = epoch
                improvement_counter += 1
                logger.info(f"  New best JSD: {best_jsd}")

        if target_metric in ('tradeoff', 'any'):
            if tradeoff > best_tradeoff:
                best_weights = model.state_dict()
                pass
                # best_user_emb = model.user_emb.weight.data.cpu().numpy()
                # best_item_emb = model.item_emb.weight.data.cpu().numpy()
                best_tradeoff = tradeoff
                last_improvement_epoch = epoch
                improvement_counter += 1
                logger.info(f"  New best tradeoff: {best_tradeoff}")

        if model_save_interval > 0 and improvement_counter >= model_save_interval:
            # if requirements defined by configs are met, save a snapshot of the embeddings
            embedding_path_dir = os.path.join(data_path, dataset, "saved_models", f'epoch_{epoch + 1}')
            os.makedirs(embedding_path_dir, exist_ok=True)
            torch.save(best_weights, os.path.join(embedding_path_dir, f'finetuned_params.pth'))

            # user_emb = model.user_emb.weight.data.cpu().numpy()
            # save_embedding(data_path, dataset, user_emb, user_id_list, user_id_column_name,
            #                user_embedding_column_name,
            #                save_path=os.path.join(embedding_path_dir, f'{dataset}.useremb'))
            # if not config['freeze_item_emb']:
            #     # Only save item embeddings if they can change
            #     item_emb = model.item_emb.weight.data.cpu().numpy()
            #     # save_embedding(data_path, dataset, item_emb, item_id_list, item_id_column_name,
            #     #                item_embedding_column_name,
            #     #                save_path=os.path.join(embedding_path_dir, f'{dataset}.itememb'))

            improvement_counter = 0

        if torch.abs(valid_ndcg - old_valid_ndcg).item() < 1.e-6 and torch.abs(jsd - old_jsd).item() < 1.e-6:
            print('  Identical result as in previous iteration encountered!')
            identical_results_counter += 1
        else:
            identical_results_counter = 0

        if identical_results_counter >= 3:
            logger.info(f"Exactly identical results encountered 3 times in a row. Assuming this is a pitfall and stopping early!")
            break

        if epoch - last_improvement_epoch >= early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    # Save the best embeddings to disk
    # save_embedding(data_path, dataset, best_user_emb, user_id_list, user_id_column_name,
    #                user_embedding_column_name, 'finetuned', 'useremb')
    # save_embedding(data_path, dataset, best_item_emb, item_id_list, item_id_column_name,
    #                item_embedding_column_name, 'finetuned', 'itememb')
    if config['model'] == 'CaliBPR':
        model = CaliBPR(base_weights_path, y_true_train, True)
    elif config['model'] == 'CaliDMF':
        model = CaliDMF(base_weights_path, y_true_train, True)
    model.load_state_dict(best_weights)
    model.to(device)

    best_valid_ndcg, best_test_ndcg, best_jsd = evaluate(config, True, False, unique_users, device,
                                                         model,
                                                         y_true_train, y_true_valid, y_true_test,
                                                         popularity_targets, item_popularity, logger)

    logger.info(64 * "#")
    logger.info(f"Baseline Validation NDCG: {init_valid_ndcg}")
    logger.info(f"Baseline Test NDCG: {init_test_ndcg}")
    logger.info(f"Baseline JSD: {init_jsd}")
    logger.info(f"-----Best model according to metric: {target_metric}----")
    logger.info(f"Finetuned Validation NDCG: {best_valid_ndcg}")
    logger.info(f"Finetuned Test NDCG: {best_test_ndcg}")
    logger.info(f"Finetuned JSD: {best_jsd}")
    logger.info(64 * '-')
    logger.info(f"Change in test NDCG: {best_test_ndcg - init_test_ndcg}")
    logger.info(f"Change in JSD: {best_jsd - init_jsd}")
    logger.info(64 * "#")


def run_calitune_from_config_file(config_path: str, dataset: str):
    config = read_config(config_path)
    run_calitune(config, dataset)