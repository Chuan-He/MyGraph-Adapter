import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import build_dataset
from datasets.utils import build_data_loader

import clip
from utils import *
import numpy as np
from gnn.models import *
from utils import *
from AGNN.sampler import *
from AGNN.few_shot import *
from AGNN.utils import *
from torch.utils.tensorboard import SummaryWriter


# def _init_(args):
#     if not os.path.exists('checkpoints'):
#         os.makedirs('checkpoints')
#     if not os.path.exists('checkpoints/'+args.exp_name):
#         os.makedirs('checkpoints/'+args.exp_name)
#     if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
#         os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
#     os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
#     os.system('cp models/models.py checkpoints' + '/' + args.exp_name + '/' + 'models.py.backup')


def get_arguments():
    # Training settings
    parser = argparse.ArgumentParser(description='Few-Shot Learning with Graph Neural Networks')
    # parser.add_argument('--exp_name', type=str, default='debug_vx', metavar='N',
    #                     help='Name of the experiment')
    parser.add_argument('--train_batches', type=int, default=100, metavar='train_batches',
                        help='batches)')
    parser.add_argument('--batch_size', type=int, default=4, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--batch_size_test', type=int, default=4, metavar='batch_size_test',
                        help='Size of batch)')
    parser.add_argument('--iterations', type=int, default=1000, metavar='iterations',
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
    parser.add_argument('--test_N_shots', type=int, default=5, metavar='test_N_shots',
                        help='Number of shots in test')
    parser.add_argument('--train_N_shots', type=int, default=5, metavar='train_N_shots',
                        help='Number of shots when training')
    parser.add_argument('--N_query', type=int, default=1, metavar='N_query',
                        help='Number of shots in test')
    parser.add_argument('--unlabeled_extra', type=int, default=0, metavar='unlabeled_extra',
                        help='Number of shots when training')
    parser.add_argument('--dataset_root', type=str, default='../DATA', metavar='dataset_root',
                        help='Root dataset')
    # parser.add_argument('--test_samples', type=int, default=30000, metavar='N',
    #                     help='Number of shots')
    parser.add_argument('--dataset', type=str, default='eurosat', metavar='dataset',
                        help='datasets')
    # parser.add_argument('--dec_lr', type=int, default=10000, metavar='N',
    #                     help='Decreasing the learning rate every x iterations')
    args = parser.parse_args()
    return args

def cal_edge_emb(x, p=2, dim=1):   # v1_graph---taking the similairty by 
    ''' 
    x: (n,K)   [m+1, 1000, 1024]
    return: (n^2, K)
    '''
    x = F.normalize(x, p=p, dim=dim)    #[m+1, 1000, 1024], [100, 1024, 101]
    x_c = x
    x = x.transpose(1, 2)  #[1000, m+1, 1024]  [100, 101, 1024]
    x_r = x  # (K, n, 1) #[1000, m+1, 1024]
    # x_c = torch.transpose(x, 1, 2)  # (K, 1, n) #[1000, 1024, m+1]
    # A = torch.bmm(x_r, x_c).permute(1,2,0)  # (n, n, K) 
    A = torch.bmm(x_c, x_r)     # [1000, m+1, m+1]

    # A = A.view(A.size(0) * A.size(1), A.size(2))  # (n^2, K)
    # print(A.size())
    return A

def _get_base_image_features(clip_model, train_loader_x):
    with torch.no_grad():
        img_feature = []
        labels = []
        for batch_idx, batch in enumerate(train_loader_x):
            image = batch["img"]
            label = batch["label"]
            image = image.cuda()
            label = label.cuda()
            image_features = clip_model.encode_image(image.type(clip_model.dtype)).detach()
            img_feature.append(image_features)
            labels.append(label)
        img_feature_list = torch.cat(img_feature, dim=0)
        label_list = torch.cat(labels, dim=0)
        sorted, indices = torch.sort(label_list)
        # print("+++++++++++++++++++len_label", len(sorted), sorted[-1])
        label_len = len(sorted)//(sorted[-1]+1)
        # print('=====label_len', label_len)
        img_feature_list_all = torch.index_select(img_feature_list, 0, indices)
        b, c = img_feature_list_all.size()
        label_list = sorted.view(b//label_len, label_len)
        img_feature_list_all = img_feature_list_all.view(b//label_len, label_len, -1).mean(dim=1)

    return img_feature_list_all




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
    
    print("Preparing dataset.")
    dataset = build_dataset(cfg['dataset'], args.dataset_root, args.train_N_shots + args.test_N_shots)

    # val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
    # test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)

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
    train_batches = n_train_way * n_train_shot
    
    trainx_label = []
    for item in dataset.train_x:
        trainx_label.append(item.label)
    train_sampler = CategoriesSampler(
            trainx_label, train_batches,
            n_train_way, n_train_shot + n_query,
            ep_per_batch=batch_size)
    train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=4, batch_sampler=train_sampler, tfm=train_tranform, is_train=True, shuffle=True)
    
    n_test_shot = args.test_N_shots
    n_test_way = args.test_N_way
    iters = args.iterations
    test_batch_size = args.batch_size_test
    test_batches = n_test_way * n_test_shot
    test_label = []
    for item in dataset.test:
        test_label.append(item.label)
    test_sampler = CategoriesSampler(
            test_label, test_batches,
            n_test_way, n_test_shot + n_query,
            ep_per_batch=test_batch_size)

    test_loader = build_data_loader(data_source=dataset.test, batch_size=4, batch_sampler=test_sampler, tfm=train_tranform, is_train=False, shuffle=True)

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    # cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache)

    # Pre-load val features
    # print("\nLoading visual features and labels from val set.")
    # val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)

    # Pre-load test features
    # print("\nLoading visual features and labels from test set.")
    # test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

    #n_classes = len(dataset.classnames)

    gnn_model = GraphNN(nfeat=1024, nhid=512, nclass=n_train_way).to('cuda')
    fusion = nn.Conv2d(2,1, kernel_size=(1,1), stride=(1,1)).to('cuda')
    softmax_module = SoftmaxModule()
    #model.float()
    # NOTE: only give prompt_learner to the optimizer
    optimizer = torch.optim.AdamW(list(gnn_model.parameters()) + list(fusion.parameters()), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iters * len(train_loader_F))

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
    max_epoch = 100
    save_epoch = 100
    max_va = 0.
    timer_used = Timer()
    timer_epoch = Timer()
    counter = 0
    total_loss = 0
    ep_per_batch = 4

    aves_keys = ['tl', 'ta', 'tvl', 'tva', 'vl', 'va']
    trlog = dict()
    for k in aves_keys:
        trlog[k] = []
    
    for epoch in range(1, max_epoch + 1):
        # train
        gnn_model.train()
        fusion.train()
        timer_epoch.s()
        aves = {k: Averager() for k in aves_keys}

        for data, _ in tqdm(train_loader_F, desc='train', leave=False):
            # print(data.size())
            x_shot, x_query = split_shot_query(
                    data.cuda(), n_train_way, n_train_shot, n_query,
                    ep_per_batch=4) # bs * n_cls * n_per

            shot_shape = x_shot.shape[:-3]
            query_shape = x_query.shape[:-3]
            img_shape = x_shot.shape[-3:]

            x_shot = x_shot.view(-1, *img_shape)
            x_query = x_query.view(-1, *img_shape)
            
            x_tot = clip_model.encode_image(torch.cat([x_shot, x_query], dim=0))
            # x_tot = self.avgpool(self.encoder(torch.cat([x_shot, x_query], dim=0))).squeeze()

            x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
            x_shot = x_shot.view(*shot_shape, -1)
            x_query = x_query.view(*query_shape, -1)

            [a1,a2,a3,a4]=x_shot.size()
            x_node = torch.cat([x_shot.reshape(a1,a2*a3,a4),x_query], dim=1)         
            [b,n,_] = x_query.size()

            label_tr = make_nk_label(n_train_way, n_train_shot,
                    ep_per_batch=ep_per_batch).cuda()
            label = make_nk_label(n_train_way, n_query,
                    ep_per_batch=ep_per_batch).cuda()

            tr_label = label_tr.unsqueeze(1)
            one_hot = torch.zeros((tr_label.size()[0],5),device='cuda')
            one_hot.scatter_(1,tr_label,1)
            one_hot_fin = one_hot.reshape(b,tr_label.size()[0]//b,5)
            # zero_pad = torch.zeros((b,n,5), device=x_query.device) # zero-init or avg-init
            zero_pad = torch.zeros((b, n, 5),device= 'cuda').fill_(1.0/5) # zero-init or avg-init
            # zero_pad.requires_grad = True
            #if self.args.cuda:
            #    zero_pad = zero_pad.cuda()
            label_fea = torch.cat([one_hot_fin,zero_pad], dim=1)
            
            # xx = torch.cat([x_node,label_fea],dim=2)
            # att = self.slf_attn(xx,xx)
            # xx = torch.bmm(att, xx)

            # #self-attention and fusion block
            x = F.normalize(x_node, p=2, dim=2, eps=1e-12)        
            x_trans = torch.transpose(x, 1, 2)  # batch, dim, N
            att = torch.bmm(x, x_trans)         # batch, N, N  (N=5*1(5)+5*15  80 or 100)
            # mask_f = F.softmax(att, dim=2)

            # att = self.slf_attn(x_node,x_node)           
            
            lab_t = torch.transpose(label_fea, 1, 2)
            att_l = torch.bmm(label_fea, lab_t)
            # mask_l = F.softmax(att_l, dim=2)

            # mask_c = torch.cat([mask_f.unsqueeze(1),mask_l.unsqueeze(1)],dim=1)
            mask_c = torch.cat([att.unsqueeze(1),att_l.unsqueeze(1)],dim=1)
            new_mask = fusion(mask_c).squeeze(1)
            # new_fea = torch.bmm(new_mask, x_node)

            new_fea = torch.bmm(new_mask, x_node.float())

            #lab_new = torch.mul(torch.bmm(new_mask,label_fea),1-self.alpha) +  torch.mul(label_fea,self.alpha)               
            xx = new_fea

            adj = cal_edge_emb(xx)

            output1, output2 = gnn_model(xx, adj)
            logits = F.sigmoid(output2)[:,25:30,:] #b*M*d

            #logits = out_fea.squeeze(-1)
            logits = logits.reshape(-1, n_train_way)

### text
            text_features = clip_weights.transpose(1,0)[0:5]
            tt = text_features.float()
            node_size = text_features.size()[0]  
            #adj = torch.clip(adj, min=0.0)
            I = torch.eye(node_size, device='cuda').to('cuda')
            tt_gcn = gnn_model.forward_text(tt, I)
            text_features = tt_gcn
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            adj = cal_edge_emb(x_node.float())
            output1, output2 = gnn_model(x_node.float(), adj)
            image_feature = output1
            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
            clip_logits = image_feature @ text_features.t()
            clip_logits = clip_logits[:,25:30,:]
            clip_logits = clip_logits.reshape(-1, n_train_way)
            
            total_logits = clip_logits + logits
            total_loss = F.cross_entropy(total_logits, label)

            acc = compute_acc(total_logits, label)
            # print(adj_gt.size())
            # print(wl.size())
            # loss2 = torch.sum(torch.norm(adj_gt-wl, dim=(1,2)))
        
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
        for data, _ in tqdm(test_loader, desc='test', leave=False):
            # test
            gnn_model.eval()
            fusion.eval()
            x_shot, x_query = split_shot_query(
                    data.cuda(), n_test_way, n_test_shot, n_query,
                    ep_per_batch=4)
            label_tr = make_nk_label(n_test_way, n_test_shot,
                    ep_per_batch=4).cuda()
            label = make_nk_label(n_test_way, n_query,
                    ep_per_batch=4).cuda()

            with torch.no_grad():
                tr_label = label_tr.unsqueeze(1)
                one_hot = torch.zeros((tr_label.size()[0],5),device='cuda')
                one_hot.scatter_(1,tr_label,1)
                one_hot_fin = one_hot.reshape(b,tr_label.size()[0]//b,5)
                # zero_pad = torch.zeros((b,n,5), device=x_query.device) # zero-init or avg-init
                zero_pad = torch.zeros((b, n, 5),device= 'cuda').fill_(1.0/5) # zero-init or avg-init
                # zero_pad.requires_grad = True
                #if self.args.cuda:
                #    zero_pad = zero_pad.cuda()
                label_fea = torch.cat([one_hot_fin,zero_pad], dim=1)
                
                # xx = torch.cat([x_node,label_fea],dim=2)
                # att = self.slf_attn(xx,xx)
                # xx = torch.bmm(att, xx)

                # #self-attention and fusion block
                x = F.normalize(x_node, p=2, dim=2, eps=1e-12)        
                x_trans = torch.transpose(x, 1, 2)  # batch, dim, N
                att = torch.bmm(x, x_trans)         # batch, N, N  (N=5*1(5)+5*15  80 or 100)
                # mask_f = F.softmax(att, dim=2)

                # att = self.slf_attn(x_node,x_node)           
                
                lab_t = torch.transpose(label_fea, 1, 2)
                att_l = torch.bmm(label_fea, lab_t)
                # mask_l = F.softmax(att_l, dim=2)

                # mask_c = torch.cat([mask_f.unsqueeze(1),mask_l.unsqueeze(1)],dim=1)
                mask_c = torch.cat([att.unsqueeze(1),att_l.unsqueeze(1)],dim=1)
                new_mask = fusion(mask_c).squeeze(1)
                # new_fea = torch.bmm(new_mask, x_node)

                new_fea = torch.bmm(new_mask, x_node.float())

                #lab_new = torch.mul(torch.bmm(new_mask,label_fea),1-self.alpha) +  torch.mul(label_fea,self.alpha)               
                xx = new_fea

                adj = cal_edge_emb(xx)

                output1, output2 = gnn_model(xx, adj)
                logits = F.sigmoid(output2)[:,25:30,:] #b*M*d

                #logits = out_fea.squeeze(-1)
                logits = logits.reshape(-1, n_train_way)
            
                text_features = clip_weights.transpose(1,0)[0:5]
                tt = text_features.float()
                node_size = text_features.size()[0]  
                #adj = torch.clip(adj, min=0.0)
                I = torch.eye(node_size, device='cuda').to('cuda')
                tt_gcn = gnn_model.forward_text(tt, I)
                text_features = tt_gcn
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                adj = cal_edge_emb(x_node.float())
                output1, output2 = gnn_model(x_node.float(), adj)
                image_feature = output1
                image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
                clip_logits = image_feature @ text_features.t()
                clip_logits = clip_logits[:,25:30,:]
                clip_logits = clip_logits.reshape(-1, n_train_way)
                
                total_logits = clip_logits + logits
                total_loss = F.cross_entropy(total_logits, label)

                acc = compute_acc(total_logits, label)

            aves['vl'].add(total_loss.sum().item())
            # aves[name_l2].add(loss2.item())
            aves['va'].add(acc)

        _sig = int(_[-1])

        # post
        for k, v in aves.items():
            aves[k] = v.item()
            trlog[k].append(aves[k])

        t_epoch = time_str(timer_epoch.t())
        t_used = time_str(timer_used.t())
        t_estimate = time_str(timer_used.t() / epoch * max_epoch)
        log('epoch {}, train {:.4f}|{:.4f}, tval {:.4f}|{:.4f}, '
        'val {:.4f}|{:.4f}, {} {}/{} (@{})'.format(
        epoch, aves['tl'], aves['ta'], aves['tvl'], aves['tva'],
        aves['vl'], aves['va'], t_epoch, t_used, t_estimate, _sig))
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