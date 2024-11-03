import math, os
from time import time
from scMSAC import scMSAC
import h5py
import torch
from utils import *
from scipy import stats, spatial
import numpy as np
import pandas as pd
from scipy import sparse as sp
import scanpy as sc
from anndata import AnnData
from scipy.spatial import distance
import dgl
import dgl.nn as dglnn


if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    import os

    cache_file_path = '/root/autodl-tmp/scMSAC/cluster_gcn.pkl'

    if os.path.exists(cache_file_path):
        os.remove(cache_file_path)

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--f1', default=2000, type=float, help='Number of mRNA after feature selection')
    parser.add_argument('--f2', default=2000, type=float, help='Number of ADT/ATAC after feature selection')
    parser.add_argument('--filter1', action='store_true', default=True, help='Do mRNA selection')
    parser.add_argument('--filter2', action='store_true', default=False, help='Do ADT/ATAC selection')
    parser.add_argument('--knn', default=20, type=int, help='K value for building KNN graph')
    parser.add_argument('--train_iter', default=200, type=int, help='number of clusters')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--encodeLayer', nargs="+", default=[128], type=int, help='encoder layer size')
    parser.add_argument('--decodeLayer1', nargs="+", default=[16,64,256], type=int, help='decoder layer size')  # [128]
    parser.add_argument('--decodeLayer2', nargs="+", default=[16,20], type=int, help='decoder layer size')
    parser.add_argument('--encodeHead', default=3, type=int, help='number of encoder heads')
    parser.add_argument('--concat', default=True, type=bool, help='concatencate or avergae multiple heads')
    parser.add_argument('--z_dim', default=32, type=int, help='dimension of latent space')
    parser.add_argument('--verbose', default=True, type=bool)
    parser.add_argument('--save_dir', default='./results')
    parser.add_argument('--final_latent_file', default=-1,
                        help='file name for the output of the embedding layer; -1 for no output')
    parser.add_argument('--final_labels', default=-1,
                        help='file name for the output of the predicted label; -1 for no output')
    parser.add_argument('--ae_weights', default=None)  #
    parser.add_argument('--weights_name', default='BMNC.pth.tar')
    parser.add_argument('--gamma', default=0.001, type=float, help='coefficient of clustering loss')
    parser.add_argument('--clustering_iters', default=200, type=int, help='iteration of clustering stage')
    parser.add_argument('--sigma1', default=2, type=float, help='noise added on data for denoising autoencoder')
    parser.add_argument('--sigma2', default=1, type=float, help='noise added on data for denoising autoencoder')
    # parser.add_argument('--run', default=1)

    args = parser.parse_args()
    print(args)
    ###read data
    # data_mat = h5py.File("/root/autodl-tmp/Multi-omics/RNA_ADT/CITESeq_GSE128639_BMNC_annodata.h5")  # CITESeq_pbmc_spector_all.h5
    # x1 = np.array(data_mat['X1'])
    # x2 = np.array(data_mat['X2'])
    # y = np.array(data_mat['Y'])
    # data_mat.close()
    data_name = 'Mimitou'
    # x1 = pd.read_csv("/root/autodl-tmp/Multi-omics/RNA_ADT/In_house_PBMC2000/RNA.csv", index_col=0).T.values
    # x2 = pd.read_csv("/root/autodl-tmp/Multi-omics/RNA_ADT/In_house_PBMC2000/ADT.csv", index_col=0).values.T
    # y = pd.read_csv( "/root/autodl-tmp/Multi-omics/RNA_ADT/In_house_PBMC2000/labels.csv", index_col=0).values.flatten()
    x1 = pd.read_csv("/root/autodl-tmp/Multi-omics/RNA_ADT/Mimitou/RNA.csv", index_col=0).T.values
    x2 = pd.read_csv("/root/autodl-tmp/Multi-omics/RNA_ADT/Mimitou/ADT.csv", index_col=0).values.T
    y = pd.read_csv("/root/autodl-tmp/Multi-omics/RNA_ADT/Mimitou/labels.csv", index_col=0)
    #
    # # x1 = pd.read_csv("/root/autodl-tmp/Multi-omics/sample/sample30/RNA_sample.csv", index_col=0).T.values
    # # x2 = pd.read_csv("/root/autodl-tmp/Multi-omics/sample/sample30/ADT_sample.csv", index_col=0).values.T
    # # y = pd.read_csv( "/root/autodl-tmp/Multi-omics/sample/sample30/label_sample.csv", index_col=0)
    # if label is 'string':
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    labels = y.apply(lambda x: label_encoder.fit_transform(x))
    labels = np.array(labels)
    y = np.reshape(labels, (-1))

    # Gene filter
    if args.filter1:
        importantGenes = geneSelection(x1, n=args.f1, plot=False)
        x1 = x1[:, importantGenes]
    if args.filter2:
        importantGenes = geneSelection(x2, n=args.f2, plot=False)
        x2 = x2[:, importantGenes]

    adata = sc.AnnData(x1)
    adata = normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True)
    x1 = adata.X.astype(np.float32)
    y1 = adata.raw.X.astype(np.float32)
    sf1 = adata.obs.size_factors

    adata1 = sc.AnnData(x2)
    adata1 = normalize(adata1, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True)
    x2 = adata1.X.astype(np.float32)
    y2 = adata1.raw.X.astype(np.float32)
    sf2 = adata1.obs.size_factors

    ###Cluster number defined by user or calculated from y (if availble)

    n_clusters = len(np.unique(y))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x1 = torch.tensor(x1)
    x1 = x1.to(torch.float32)
    x2 = torch.tensor(x2)
    x2 = x2.to(torch.float32)
    y1 = torch.tensor(y1).to(torch.float32)
    y2 = torch.tensor(y2).to(torch.float32)
    count = torch.cat([x1, x2], dim=-1)  # .to(device)
    #############################################################################################
    print("***Building graph***")
    adj = pd.read_csv('/root/autodl-tmp/Multi-omics/RNA_ADT/Mimitou/wnn_adj.csv', index_col=0).values    #10X10k
    #
    # adj, _ = get_adj(x1, k=20, pca=30)  #
    # adj = getGraph(count,'Inhouse', L=0, K=20, method='spearman')
    adj_ = sp.csr_matrix(adj)
    adj_n = norm_adj(adj_)

    size_factor1 = torch.tensor(sf1, dtype=torch.float32)
    size_factor2 = torch.tensor(sf2, dtype=torch.float32)
    #X = torch.tensor(count, dtype=torch.float32)  # .to(self.device)
    G = dgl.from_scipy(adj_n)
    G.ndata['x1'] = x1
    G.ndata['x2'] = x2
    G.ndata['X_raw1'] = y1
    G.ndata['X_raw2'] = y2
    G.ndata['size_factor1'] = size_factor1
    G.ndata['size_factor2'] = size_factor2
    num_partitions =200
    sampler = dgl.dataloading.ClusterGCNSampler(
        G,
        num_partitions,
        prefetch_ndata=["x1", "x2", "X_raw1", "X_raw2", "size_factor1", "size_factor2"],
    )
    dataloader = dgl.dataloading.DataLoader(
        G,
        torch.arange(num_partitions).to("cuda"),
        sampler,
        device="cuda",
        batch_size=100,
        shuffle=True,
        drop_last=False,
        use_uva=True
    )

    if x1.shape[0] > 8000:
        thoo = 0
    else:
        thoo = 1

    ###build model
    model = scMSAC(input_dim1=adata.n_vars, input_dim2=adata1.n_vars, encodeLayer=args.encodeLayer,
                   decodeLayer1=args.decodeLayer1, decodeLayer2=args.decodeLayer2, encodeHead=args.encodeHead,
                   encodeConcat=args.concat, beta=0.1, gamma=args.gamma, activation="elu", z_dim=args.z_dim, sigma1=args.sigma1,
                   sigma2=args.sigma2, tho = thoo, device=device).to(device)

    print(str(model))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    t0 = time()

    ###pretraining stage
    if args.ae_weights is None:
        # model.train_model(count, A_n=adj_n, A=adj,X_raw1=y1, X_raw2=y2, size_factor1=sf1,size_factor2=sf2, lr=args.lr, train_iter=args.train_iter, verbose=args.verbose)
        model.train_model(x1, x2, G=G,dataloader=dataloader,
                          lr=args.lr,
                          train_iter=args.train_iter, verbose=args.verbose, weights_name=args.weights_name)
    else:
        if os.path.isfile(args.ae_weights):
            print("==> loading checkpoint '{}'".format(args.ae_weights))
            checkpoint = torch.load(args.ae_weights)
            model.load_state_dict(checkpoint['ae_state_dict'])
        else:
            print("==> no checkpoint found at '{}'".format(args.ae_weights))
            raise ValueError
    print('Pret-raining time: %d seconds.' % int(time() - t0))

    ###clustering stage
    y_pred, Zdata,final_loss, epoch, z3, z4 = model.fit(x1, x2, G=G,dataloader=dataloader,
                                          n_clusters=n_clusters, num_epochs=args.clustering_iters, y=y,
                                          lr=0.8, update_interval=1, tol=0.001)

    t1 = time()
    print("Time used is:" + str(t1 - t0))

    ###evaluation if y is available
    acc, nmi, ari = eval_cluster(y, y_pred)

    print('Final Clustering: ACC= %.4f, NMI= %.4f, ARI= %.4f, Loss = %.8f' % (acc, nmi, ari, final_loss))

    # final_latent = z3.detach().cpu().numpy()
    # final_latent = pd.DataFrame(final_latent)
    # final_latent.to_csv(args.save_dir + "/" + data_name + "_rna.csv", index=True, header=True)
    # final_latent = z4.detach().cpu().numpy()
    # final_latent = pd.DataFrame(final_latent)
    # final_latent.to_csv(args.save_dir + "/" + data_name + "_adt.csv", index=True, header=True)

    ###output predicted labels and embedding
    if args.final_latent_file != -1:
        final_latent = Zdata.cpu().numpy()
        # np.savetxt(args.save_dir + "/" + args.final_latent_file + "_" + str(args.run), final_latent, delimiter=",")
        final_latent = pd.DataFrame(final_latent)
        final_latent.to_csv(args.save_dir + "/" + data_name + "_embedding.csv", index=True, header=True)
    if args.final_labels != -1:
        # np.savetxt(args.save_dir + "/" + args.final_labels + "_pre.csv", y_pred, delimiter=",")
        pre = pd.DataFrame(y_pred)
        pre.to_csv(args.save_dir + "/" + data_name + "_pre.csv", index=True, header=True)

