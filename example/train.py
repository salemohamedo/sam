import argparse
import torch

from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
import wandb

import sys; sys.path.append("..")
from sam import SAM

def pairwise_cosine_sim(a, b, eps=1e-8, norm=True):
    """
    ### Taken from https://stackoverflow.com/questions/50411191/how-to-compute-the-cosine-similarity-in-pytorch-for-all-rows-in-a-matrix-with-re
    added eps for numerical stability
    """
    if norm == False:
        return torch.mm(a, b.transpose(0, 1))
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def sam_feat_loss(z, y):
    # Compute pairwise cosine similarity
    similarity_matrix = pairwise_cosine_sim(z, z)
    # similarity_matrix = torch.exp(similarity_matrix)
    
    # Get the mask for positive pairs (same class) and negative pairs (different classes)
    positive_mask = y.unsqueeze(1) == y.unsqueeze(0)
    negative_mask = ~positive_mask

    # We might want to ignore the diagonal (self-similarity) so we set it to zero
    eye = torch.eye(positive_mask.size(0), device=positive_mask.device).bool()
    positive_mask[eye] = False

    # Compute the loss for positive and negative pairs
    positive_similarity = similarity_matrix[positive_mask].mean()
    negative_similarity = similarity_matrix[negative_mask].mean()

    # The loss aims to maximize positive similarity and minimize negative similarity
    # You can introduce a margin or scaling factor as needed

    return positive_similarity, negative_similarity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=400, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=0.05, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--sam_loss", default='ce', type=str, choices=['ce', 'feat_cos_sim', 'feat_cos_sim_pos'])
    parser.add_argument("--sam_feat_alpha", default=1.0, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--wandb_name", default='', type=str)
    parser.add_argument("--wandb_tag", default=None, type=str)
    parser.add_argument("--use_wandb", default=False, type=bool)
    args = parser.parse_args()

    initialize(args, seed=args.seed)
    device = torch.device("cuda:0")
    
    if args.use_wandb:
            wandb.init(
            entity='mila-projects', 
            project='experiments', 
            dir='/network/scratch/o/omar.salemohamed/wandb', 
            name=args.wandb_name, 
            tags=[args.wandb_tag] if args.wandb_tag != None else [], 
            config=args
            )

    dataset = Cifar(args.batch_size, args.threads)
    log = Log(log_each=10)
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)
    hook = model.f[-1].register_forward_hook(hook=lambda m, i, o: (i[0], o))  

    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))

        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)

            # first forward-backward step
            enable_running_stats(model)
            feats, logits = model(inputs)
            loss = smooth_crossentropy(logits, targets, smoothing=args.label_smoothing)
            if args.sam_loss == 'ce':
                loss.mean().backward()
            elif args.sam_loss == 'feat_cos_sim_pos':
                # print('here')
                pos_sim, neg_sim = sam_feat_loss(feats, targets)
                sam_loss = -pos_sim
                sam_loss.backward()
            else:
                pos_sim, neg_sim = sam_feat_loss(feats, targets)
                sam_loss = -pos_sim + args.sam_feat_alpha*neg_sim
                sam_loss.backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward step
            disable_running_stats(model)
            smooth_crossentropy(model(inputs)[1], targets, smoothing=args.label_smoothing).mean().backward()
            optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                correct = torch.argmax(logits.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)

        train_loss, train_acc = log.flush(print_stats=False)
        model.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)

                feats, logits = model(inputs)
                loss = smooth_crossentropy(logits, targets)
                correct = torch.argmax(logits, 1) == targets
                log(model, loss.cpu(), correct.cpu())
        
        test_loss, test_acc = log.flush(print_stats=False)
        if args.use_wandb:
            wandb.log({
                'train_loss': train_loss,
                'test_loss': test_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'epoch': epoch,
            }, step=epoch)


    log.flush()
