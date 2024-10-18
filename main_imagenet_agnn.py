import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 0, 2, 3'
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
# import numpy

from datasets import build_dataset
from datasets.utils import build_data_loader, build_data_loader1

import clip
from utils import *
import numpy as np
from gnn.models import *
from utils import *
from AGNN.sampler import *
from AGNN.few_shot import *
from AGNN.utils import *
from torch.utils.tensorboard import SummaryWriter

from datasets.imagenet import ImageNet


def get_arguments():
    # Training settings
    parser = argparse.ArgumentParser(description='Few-Shot Learning with Graph Neural Networks')
    # parser.add_argument('--exp_name', type=str, default='debug_vx', metavar='N',
    #                     help='Name of the experiment')
    # parser.add_argument('--train_batches', type=int, default=100, metavar='train_batches',
    #                     help='batches)')
    parser.add_argument('--batch_size', type=int, default=4, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--batch_size_test', type=int, default=16, metavar='batch_size_test',
                        help='Size of batch)')
    parser.add_argument('--iterations', type=int, default=200, metavar='iterations',
                        help='number of epochs to train ')
    # parser.add_argument('--decay_interval', type=int, default=10000, metavar='N',
    #                     help='Learning rate decay interval')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    # parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
    #                     help='SGD momentum (default: 0.5)')
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='seed',
                        help='random seed (default: 1)')
    # parser.add_argument('--log-interval', type=int, default=20, metavar='N',
    #                     help='how many batches to wait before logging training status')
    # parser.add_argument('--save_interval', type=int, default=300000, metavar='N',
    #                     help='how many batches between each model saving')
    # parser.add_argument('--test_interval', type=int, default=2000, metavar='N',
    #                     help='how many batches between each test')
    # parser.add_argument('--test_N_way', type=int, default=5, metavar='N',
    #                     help='Number of classes for doing each classification run')
    parser.add_argument('--train_N_way', type=int, default=5, metavar='train_N_way',
                         help='Number of classes for doing each training comparison')
    parser.add_argument('--test_N_way', type=int, default=5, metavar='test_N_way',
                         help='Number of classes for doing each testing comparison')
    parser.add_argument('--test_N_shots', type=int, default=1, metavar='test_N_shots',
                        help='Number of shots in test')
    parser.add_argument('--train_N_shots', type=int, default=1, metavar='train_N_shots',
                        help='Number of shots when training')
    parser.add_argument('--N_query', type=int, default=1, metavar='N_query',
                        help='Number of shots in test')
    parser.add_argument('--unlabeled_extra', type=int, default=0, metavar='unlabeled_extra',
                        help='Number of shots when training')
    parser.add_argument('--dataset_root', type=str, default='../DATA', metavar='dataset_root',
                        help='Root dataset')
    # parser.add_argument('--test_samples', type=int, default=30000, metavar='N',
    #                     help='Number of shots')
    parser.add_argument('--dataset', type=str, default='imagenet', metavar='dataset',
                        help='datasets')
    # parser.add_argument('--dec_lr', type=int, default=10000, metavar='N',
    #                     help='Decreasing the learning rate every x iterations')
    args = parser.parse_args()
    return args


def main():

    # _init_()
    # Load config file
    args = get_arguments()
  
    cfg = yaml.load(open('configs/' + args.dataset + '.yaml', 'r'), Loader=yaml.Loader)

    svname = 'meta_{}-{}shot'.format(args.dataset, args.train_N_shots)
    save_path = os.path.join('./save', svname)

#   cp_exist = utils.ensure_path(save_path)
    ensure_path(save_path)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # Prepare dataset
    random.seed(1)
    torch.manual_seed(1)

    print("Preparing ImageNet dataset.")
    imagenet = ImageNet(args.dataset_root, args.train_N_shots, preprocess)

    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    n_train_shot = args.train_N_shots
    n_query = args.N_query
    n_train_way = args.train_N_way
    iters = args.iterations
    batch_size = args.batch_size
    train_batches = 20
    
    trainx_label = []
    for item in imagenet.train:
        trainx_label.append(item[1])
    train_sampler = CategoriesSampler(
            trainx_label, train_batches,
            1, n_query,
            ep_per_batch=batch_size)
    

    train_loader_F = torch.utils.data.DataLoader(imagenet.train, batch_size=n_train_shot, num_workers=8, shuffle=False)
    train_loader_sample = torch.utils.data.DataLoader(imagenet.train, batch_sampler=train_sampler, num_workers=8)

    n_test_shot = args.test_N_shots
    n_test_way = args.test_N_way
    iters = args.iterations
    test_batch_size = args.batch_size_test
    test_batches = 20
    test_label = []
    for item in imagenet.test:
        test_label.append(item[1])
    test_sampler = CategoriesSampler(
            test_label, test_batches,
            1, n_query,
            ep_per_batch=test_batch_size)

    #test_loader = build_data_loader1(data_source=imagenet.test, batch_sampler=test_sampler, tfm=preprocess, is_train=False)
    test_loader = torch.utils.data.DataLoader(imagenet.test, batch_sampler=test_sampler, num_workers=8)

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(imagenet.classnames, imagenet.template, clip_model)

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    # cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache)

    # Pre-load val features
    # print("\nLoading visual features and labels from val set.")
    # val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)

    # Pre-load test features
    # print("\nLoading visual features and labels from test set.")
    # test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

    n_classes = len(imagenet.classnames)

    gnn_model = GraphNN(nfeat=512).cuda()
    gnn_model_text = GraphNN(nfeat=512).cuda()
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        gnn_model = nn.DataParallel(gnn_model, device_ids=[0, 1, 2])
        # gnn_model_text = nn.DataParallel(gnn_model_text, device_ids=[0, 1, 2])
    #fusion = nn.Conv2d(2,1, kernel_size=(1,1), stride=(1,1)).to('cuda')
    # softmax_module = SoftmaxModule()
    #model.float()
    # NOTE: only give prompt_learner to the optimizer
    optimizer = torch.optim.AdamW(gnn_model.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iters * len(train_loader_sample))

    # ------------------------------------------ Tip-Adapter ------------------------------------------
    print("\n-------- Searching hyperparameters on the val set. --------")
    ####################
    # Display
    ####################
    # counter += 1
    # total_loss += loss.item()
    # display_str = 'Train Iter: {}'.format(epoch)
    # display_str += '\tLoss_d_metric: {:.6f}'.format(total_loss/counter)
    total_loss = 0
    max_epoch = 200
    save_epoch = 200
    max_va = 0.
    timer_used = Timer()
    timer_epoch = Timer()
    counter = 0
    total_loss = 0
    ep_per_batch = batch_size

    aves_keys = ['tl', 'ta', 'tvl', 'tva', 'vl', 'va']
    trlog = dict()
    for k in aves_keys:
        trlog[k] = []
    
    with torch.no_grad():
        img_feature = []
        labels = []
        for batch_idx, batch in enumerate(train_loader_F):
            image = batch[0]
            label = batch[1]
            image = image.cuda()
            label = label.cuda()
            image_features = clip_model.encode_image(image.type(clip_model.dtype)).detach()
            img_feature.append(torch.mean(image_features, dim=0, keepdim=True))
            labels.append(label[0])
        sup_nodes = torch.cat(img_feature, dim=0) # n_cls * feature_dim
        label_sup = torch.stack(labels, dim=0) # n_cls
        # sorted, indices = torch.sort(label_list)
        # print("+++++++++++++++++++len_label", len(sorted), sorted[-1])
        # label_len = len(sorted)//(sorted[-1]+1)
        # print('=====label_len', label_len)
        # img_feature_list_all = torch.index_select(img_feature_list, 0, indices)

    for epoch in range(1, max_epoch + 1):
        # train
        gnn_model.train()
        #gnn_model_text.train()
        timer_epoch.s()
        aves = {k: Averager() for k in aves_keys}

        for data, label_train in tqdm(train_loader_sample, desc='train', leave=False): # 4, 5, 6   120

            with torch.no_grad():
                img_shape = data.shape[1:]
                x_query = data.view(ep_per_batch * n_query, *img_shape).cuda()
                x_query = clip_model.encode_image(x_query)

                [b, dim] = x_query.size()

                x_node = torch.cat([sup_nodes.unsqueeze(0).expand(b, -1, -1), x_query.unsqueeze(1)], dim=1)
                xx = x_node.float().detach()
                #adj = cal_edge_emb(xx).detach()

            x_gcn1 = gnn_model(xx)

            ### text
            with torch.no_grad():
                text_features = clip_weights.transpose(0,1)
                t_nodes = text_features.float().detach()
                tt = torch.cat([t_nodes.unsqueeze(0).expand(b, -1, -1), x_query.unsqueeze(1)], dim=1)
                tt = tt.float().detach()
            t_gcn1 = gnn_model(tt)

            x_gcn1 = x_gcn1 / x_gcn1.norm(dim=-1, keepdim=True)
            xgcn1_sup = x_gcn1[:,:-1,:] # 4, 10, 1024
            xgcn1_query = x_gcn1[:,-1,:] # 4, 1, 1024
            # xgcn1_sup = xgcn1_sup / xgcn1_sup.norm(dim=-1, keepdim=True)
            # xgcn1_query = xgcn1_query / xgcn1_query.norm(dim=-1, keepdim=True)

            t_gcn1 = t_gcn1 / t_gcn1.norm(dim=-1, keepdim=True)
            tgcn1_sup = t_gcn1[:,:-1,:] # 4, 10, 1024
            tgcn1_query = t_gcn1[:,-1,:] # 4, 1, 1024
            # tgcn1_sup = tgcn1_sup / tgcn1_sup.norm(dim=-1, keepdim=True)
            # tgcn1_query = tgcn1_query / tgcn1_query.norm(dim=-1, keepdim=True)
            
            logits_q1 = clip_model.logit_scale * torch.bmm(xgcn1_query.unsqueeze(1), xgcn1_sup.permute(0, 2, 1))
            logits_q2 = clip_model.logit_scale * torch.bmm(tgcn1_query.unsqueeze(1), tgcn1_sup.permute(0, 2, 1))
            logits_ce = ((logits_q1 + logits_q2) / 2).squeeze(1)
            loss_ce = F.cross_entropy(logits_ce, label_train.cuda())

            # semantic_loss2 = F.kl_div(
            #     F.log_softmax(logits_q2 / 1, dim=1),
            #     F.log_softmax(logits_q1 / 1, dim=1),
            #     reduction='sum',
            #     log_target=True
            # ) * (1 * 1) / logits_q2.numel()
            # semantic_loss1 = F.kl_div(
            #     F.log_softmax(logits_q1 / 1, dim=1),
            #     F.log_softmax(logits_q2 / 1, dim=1),
            #     reduction='sum',
            #     log_target=True
            # ) * (1 * 1) / logits_q1.numel()
            # semantic_loss2 = F.kl_div(
            #     F.log_softmax(logits_q2 / 1, dim=1),
            #     F.softmax(logits_q1 / 1, dim=1),
            #     reduction='mean') 
            # semantic_loss1 = F.kl_div(
            #     F.log_softmax(logits_q1 / 1, dim=1),
            #     F.softmax(logits_q2 / 1, dim=1),
            #     reduction='mean')
            total_loss = loss_ce #+ 500. * (semantic_loss1 + semantic_loss2)

            acc = compute_acc(logits_ce, label_train.cuda())
        
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            # aves['tl2'].add(loss2.item())
            aves['tl'].add(total_loss.sum().item())
            aves['ta'].add(acc)

            logits = None; total_loss = None; loss = None; #loss2 = None
        ####################
        # Test
        ####################
        ep_per_batch = test_batch_size

        for data, label_test in tqdm(test_loader, desc='test', leave=False):
            # test
            gnn_model.eval()
            #gnn_model_text.eval()
            with torch.no_grad():
                img_shape = data.shape[1:]
                x_query = data.view(ep_per_batch * n_query, *img_shape).cuda()
                x_query = clip_model.encode_image(x_query)
                # x_tot = self.avgpool(self.encoder(torch.cat([x_shot, x_query], dim=0))).squeeze()

                [b, dim] = x_query.size()

                x_node = torch.cat([sup_nodes.unsqueeze(0).expand(b, -1, -1), x_query.unsqueeze(1)], dim=1)
                xx = x_node.float().detach()
                #adj = cal_edge_emb(xx).detach()

                x_gcn1 = gnn_model(xx)

            ### text

                text_features = clip_weights.transpose(0,1)
                t_nodes = text_features.float().detach()
                tt = torch.cat([t_nodes.unsqueeze(0).expand(b, -1, -1), x_query.unsqueeze(1)], dim=1)
                tt = tt.float().detach()
                #adj = cal_similarity(tt).detach()
                # node_size = text_features.size()[0]
                #adj = torch.eye(node_size, device='cuda').to('cuda')
                t_gcn1 = gnn_model(tt)

                x_gcn1 = x_gcn1 / x_gcn1.norm(dim=-1, keepdim=True)
                xgcn1_sup = x_gcn1[:,:-1,:] # 4, 10, 1024
                xgcn1_query = x_gcn1[:,-1,:] # 4, 1, 1024

                t_gcn1 = t_gcn1 / t_gcn1.norm(dim=-1, keepdim=True)
                tgcn1_sup = t_gcn1[:,:-1,:] # 4, 10, 1024
                tgcn1_query = t_gcn1[:,-1,:] # 4, 1, 1024
                
                logits_q1 = clip_model.logit_scale * torch.bmm(xgcn1_query.unsqueeze(1), xgcn1_sup.permute(0, 2, 1))
                logits_q2 = clip_model.logit_scale * torch.bmm(tgcn1_query.unsqueeze(1), tgcn1_sup.permute(0, 2, 1))
                logits_ce = ((logits_q1 + logits_q2) / 2).squeeze(1)
                loss_ce = F.cross_entropy(logits_ce, label_test.cuda())

                acc = compute_acc(logits_ce, label_test.cuda())
                aves['vl'].add(loss_ce.sum().item())
                aves['va'].add(acc)

        # _sig = int(_[-1])

        # post
        for k, v in aves.items():
            aves[k] = v.item()
            trlog[k].append(aves[k])

        t_epoch = time_str(timer_epoch.t())
        t_used = time_str(timer_used.t())
        t_estimate = time_str(timer_used.t() / epoch * max_epoch)
        log('epoch {}, train {:.4f}|{:.4f}, tval {:.4f}|{:.4f}, '
        'val {:.4f}|{:.4f}, {} {}/{}'.format(
        epoch, aves['tl'], aves['ta'], aves['tvl'], aves['tva'],
        aves['vl'], aves['va'], t_epoch, t_used, t_estimate))
        # utils.log('epoch {}, train {:.4f},{:.4f}|{:.4f}, tval {:.4f},{:.4f}|{:.4f}, '
        #         'val {:.4f},{:.4f}|{:.4f}, {} {}/{} (@{})'.format(
        #         epoch, aves['tl'], aves['tl2'], aves['ta'], aves['tvl'], aves['tvl2'], aves['tva'],
        #         aves['vl'], aves['vl2'], aves['va'], t_epoch, t_used, t_estimate, _sig))

        writer.add_scalars('loss', {
            'train': aves['tl'],
            'tval': aves['tvl'],
            'val': aves['vl'],
            # 'train2': aves['tl2'],
            # 'tval2': aves['tvl2'],
            # 'val2': aves['vl2'],            
        }, epoch)
        writer.add_scalars('acc', {
            'train': aves['ta'],
            'tval': aves['tva'],
            'val': aves['va'],
        }, epoch)

        training = {
            'epoch': epoch,
            # 'optimizer': config['optimizer'],
            # 'optimizer_args': config['optimizer_args'],
            'optimizer_sd': optimizer.state_dict(),
        }
        save_obj = {
            'file': __file__,
            # 'config': config,

            # 'model': config['model'],
            # 'model_args': config['model_args'],
            'model_gnn': gnn_model.state_dict(),

            'training': training,
        }
        torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))
        torch.save(trlog, os.path.join(save_path, 'trlog.pth'))

        if (save_epoch is not None) and epoch % save_epoch == 0:
            torch.save(save_obj,
                    os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if aves['va'] > max_va:
            max_va = aves['va']
            torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))

        writer.flush()
           

if __name__ == '__main__':
    main()
