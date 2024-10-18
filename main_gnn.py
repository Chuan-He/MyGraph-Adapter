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
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--batch_size_test', type=int, default=100, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--iterations', type=int, default=1000, metavar='N',
                        help='number of epochs to train ')
    # parser.add_argument('--decay_interval', type=int, default=10000, metavar='N',
    #                     help='Learning rate decay interval')
    # parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
    #                     help='learning rate (default: 0.01)')
    # parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
    #                     help='SGD momentum (default: 0.5)')
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # parser.add_argument('--log-interval', type=int, default=20, metavar='N',
    #                     help='how many batches to wait before logging training status')
    # parser.add_argument('--save_interval', type=int, default=300000, metavar='N',
    #                     help='how many batches between each model saving')
    # parser.add_argument('--test_interval', type=int, default=2000, metavar='N',
    #                     help='how many batches between each test')
    # parser.add_argument('--test_N_way', type=int, default=5, metavar='N',
    #                     help='Number of classes for doing each classification run')
    # parser.add_argument('--train_N_way', type=int, default=5, metavar='N',
    #                     help='Number of classes for doing each training comparison')
    parser.add_argument('--test_N_shots', type=int, default=1, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--train_N_shots', type=int, default=1, metavar='N',
                        help='Number of shots when training')
    parser.add_argument('--unlabeled_extra', type=int, default=0, metavar='N',
                        help='Number of shots when training')
    parser.add_argument('--dataset_root', type=str, default='../DATA', metavar='N',
                        help='Root dataset')
    # parser.add_argument('--test_samples', type=int, default=30000, metavar='N',
    #                     help='Number of shots')
    parser.add_argument('--dataset', type=str, default='eurosat', metavar='N',
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
    A = torch.bmm(x_r, x_c)     # [1000, m+1, m+1]

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

    io = IOStream('run.log')
    io.cprint(str(args))
    
    cfg = yaml.load(open('configs/' + args.dataset + '.yaml', 'r'), Loader=yaml.Loader)

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
    dataset = build_dataset(cfg['dataset'], args.dataset_root, args.train_N_shots)

    val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)

    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    #train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)
    train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=16, tfm=train_tranform, is_train=True, shuffle=True)

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

    n_classes = len(dataset.classnames)

    metric_nn = MetricNN(n_classes)
    softmax_module = SoftmaxModule()

    # ------------------------------------------ Tip-Adapter ------------------------------------------
    print("\n-------- Searching hyperparameters on the val set. --------")

    counter = 0
    total_loss = 0
    val_acc, val_acc_aux = 0, 0
    test_acc = 0
    for batch_idx in range(100):

        ####################
        # Train
        ####################
        batch_x, label_x, batches_xi, labels_yi, _, _ = \
        dataset.get_task_batch(data_source=dataset.train_x, batch_size=16, n_way=5, num_shots=1, tfm=train_tranform)

        # Compute embedding from x and xi_s
        z = clip_model.encode_image(batch_x)
        zi_s = [clip_model.encode_image(batch_xi) for batch_xi in batches_xi]

        # Compute metric from embeddings
        out_logits, gcn_feats = metric_nn(z, zi_s, labels_yi)
        logsoft_prob = softmax_module.forward(out_logits)

        # Loss
        label_x_numpy = label_x.cpu().data.numpy()
        formatted_label_x = np.argmax(label_x_numpy, axis=1)
        formatted_label_x = Variable(torch.LongTensor(formatted_label_x))
        formatted_label_x = formatted_label_x.cuda()
        loss = F.nll_loss(logsoft_prob, formatted_label_x)

        with torch.no_grad():
            img_feature = []
            labels = []
            for batch_idx, batch in enumerate(train_loader_F):
                image = batch[0]
                label = batch[1]
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
            node_cluster_i = img_feature_list_all.view(b//label_len, label_len, -1).mean(dim=1)
           
        graph_o_t_all = []
            

        # print("========index", index)
        with torch.no_grad():
            inputs_text = clip_weights.unsqueeze(dim=1)    #[100, 1, 1024]
            inputs_img = img_feature.unsqueeze(dim=1)
            node_cluster_it =  node_cluster_i[:, :, :].repeat(inputs_text.size()[0], 1, 1)  # i -> t
            feat_it = torch.cat([inputs_text, node_cluster_it], dim=1)
            feat_it = feat_it.transpose(1, 2).detach()
            edge_it = cal_edge_emb(feat_it).detach()
        graph_o_it = metric_nn(feat_it, edge_it)
        graph_o_t = (graph_o_it)*0.1 + (1-0.1)*graph_o_it
        graph_o_t_all.append(graph_o_t)

        loss.backward()


        ####################
        # Display
        ####################
        counter += 1
        total_loss += loss.item()
        display_str = 'Train Iter: {}'.format(batch_idx)
        display_str += '\tLoss_d_metric: {:.6f}'.format(total_loss/counter)
        counter = 0
        total_loss = 0

        ####################
        # Test
        ####################
        metric_nn.eval()
        correct = 0
        total = 0
        #iterations = int(test_samples/args.batch_size_test)
        for i in range(100):
            batch_x, label_x, batches_xi, labels_yi, _, _ = \
            dataset.get_task_batch(data_source=dataset.test, batch_size=16, n_way=5, num_shots=1, tfm=preprocess)

            if args.cuda:
                xi_s = [batch_xi.cuda() for batch_xi in xi_s]
                labels_yi = [label_yi.cuda() for label_yi in labels_yi]
                oracles_yi = [oracle_yi.cuda() for oracle_yi in oracles_yi]
                hidden_labels = hidden_labels.cuda()
                x = x.cuda()
            else:
                labels_yi = labels_yi

            xi_s = [Variable(batch_xi) for batch_xi in xi_s]
            labels_yi = [Variable(label_yi) for label_yi in labels_yi]
            oracles_yi = [Variable(oracle_yi) for oracle_yi in oracles_yi]
            hidden_labels = Variable(hidden_labels)
            x = Variable(x)

            # Compute embedding from x and xi_s
            z = clip_model.encode_image(batch_x)[-1]
            zi_s = [clip_model.encode_image(batch_xi)[-1] for batch_xi in batches_xi]

            # Compute metric from embeddings
            output, out_logits = metric_nn(inputs=[z, zi_s, labels_yi, oracles_yi, hidden_labels])
            output = out_logits
            y_pred = softmax_module.forward(output)
            y_pred = y_pred.data.cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)
            labels_x_cpu = labels_x_cpu.numpy()
            labels_x_cpu = np.argmax(labels_x_cpu, axis=1)

            for row_i in range(y_pred.shape[0]):
                if y_pred[row_i] == labels_x_cpu[row_i]:
                    correct += 1
                total += 1

            if (i+1) % 100 == 0:
                io.cprint('{} correct from {} \tAccuracy: {:.3f}%)'.format(correct, total, 100.0*correct/total))

        io.cprint('{} correct from {} \tAccuracy: {:.3f}%)'.format(correct, total, 100.0*correct/total))
        io.cprint('*** TEST FINISHED ***\n'.format(correct, total, 100.0 * correct / total))
        metric_nn.train()
           

if __name__ == '__main__':
    main()