import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from layers import DecoderBN, ConditionalDecoderBN, NBLoss, KLLoss, InnerProduct, ZINBLoss, MeanAct, DispAct
import numpy as np
from dgl.nn.pytorch.conv import GraphConv, SAGEConv, GATConv
import dgl
import networkx as nx
import math
from utils import *
from sklearn.cluster import KMeans
from attention import *

eps = 1e-10
MAX_LOGVAR = 15
import time
from torch.nn import Parameter

def buildNetwork(layers, activation="relu", dropout=0.):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        net.append(nn.BatchNorm1d(layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        elif activation=="elu":
            net.append(nn.ELU())
        elif activation=="lrelu":
            net.append(nn.LeakyReLU(negative_slope=0.2))
        if dropout > 0:
            net.append(nn.Dropout(p=dropout))
    return nn.Sequential(*net)



class GATEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, num_heads=1, dropout=0, concat=False, residual=False):
        super(GATEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_feats=input_dim, out_feats=hidden_dims[0], num_heads=num_heads, feat_drop=dropout, attn_drop=dropout, residual=residual, activation=F.elu, allow_zero_in_degree=True))

        if concat:
            self.layers.append(nn.BatchNorm1d(hidden_dims[0]*num_heads))
            self.layers.append(nn.ELU())
            for i in range(1, len(hidden_dims)):
                self.layers.append(GATConv(in_feats=hidden_dims[i-1]*num_heads, out_feats=hidden_dims[i], num_heads=num_heads, feat_drop=dropout, attn_drop=dropout, residual=residual, activation=F.elu, allow_zero_in_degree=True))
                self.layers.append(nn.BatchNorm1d(hidden_dims[i]*num_heads))
                self.layers.append(nn.ELU())
            self.enc_mu = GATConv(in_feats=hidden_dims[-1]*num_heads, out_feats=output_dim, num_heads=num_heads, feat_drop=0, attn_drop=0, residual=residual, activation=None, allow_zero_in_degree=True)
        else:
            self.layers.append(nn.BatchNorm1d(hidden_dims[0]))
            self.layers.append(nn.ELU())
            for i in range(1, len(hidden_dims)):
                self.layers.append(GATConv(in_feats=hidden_dims[i-1], out_feats=hidden_dims[i], num_heads=num_heads, feat_drop=dropout, attn_drop=dropout, residual=residual, activation=F.elu, allow_zero_in_degree=True))
                self.layers.append(nn.BatchNorm1d(hidden_dims[i]))
                self.layers.append(nn.ELU())
            self.enc_mu = GATConv(in_feats=hidden_dims[-1], out_feats=output_dim, num_heads=num_heads, feat_drop=0, attn_drop=0, residual=residual, activation=None, allow_zero_in_degree=True)
        self.dropout = dropout
        self.concat = concat
        self.hidden_dims = hidden_dims

    def forward(self, g, x):
        if self.concat:
            for i in range(0, len(self.hidden_dims)):
                x = self.layers[3*i](g, x)
                x = x.view(x.shape[0], x.shape[1]*x.shape[2])
                x = self.layers[3*i+1](x)
                x = self.layers[3*i+2](x)
        else:
            for i in range(0, len(self.hidden_dims)):
                x = self.layers[3*i](g, x)
                x = torch.sum(x, dim=1)
                x = self.layers[3*i+1](x)
                x = self.layers[3*i+1](x)
        mean = torch.sum(self.enc_mu(g, x), dim=1)
        return mean

class scMSAC(nn.Module):
    def __init__(self, input_dim1, input_dim2, encodeLayer=[], decodeLayer1=[], decodeLayer2=[], encodeHead=3,
                 encodeConcat=False,
                 activation='elu', z_dim=32, alpha=1., beta=0.1, gamma=0.1, sigma1=2.5, sigma2=1.5, tho=1, device="cuda"):
        super(scMSAC, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.encodeLayer = encodeLayer
        self.decodeLayer1 = decodeLayer1
        self.decodeLayer2 = decodeLayer2
        self.encodeConcat = encodeConcat
        self.z_dim = z_dim
        self.tau = 1.
        self.tho = tho
        self.num_encoderLayer = len(encodeLayer) + 1
        self.activation = activation
        # self.encoder = SAGE(in_feats=input_dim1 + input_dim2, n_hidden=encodeLayer, n_classes=z_dim)
        self.encoder = GATEncoder(input_dim=input_dim1, hidden_dims=encodeLayer, output_dim=z_dim,
                                  num_heads=encodeHead)
        self.encoder2 = GATEncoder(input_dim=input_dim2, hidden_dims=encodeLayer, output_dim=z_dim,
                                  num_heads=encodeHead)
        self.decoder1 = buildNetwork([z_dim] + decodeLayer1, activation=activation, dropout=0.)
        self.decoder2 = buildNetwork([z_dim] + decodeLayer2, activation=activation, dropout=0.)
        self.dec_mean1 = nn.Sequential(nn.Linear(decodeLayer1[-1], input_dim1), MeanAct())
        self.dec_disp1 = nn.Sequential(nn.Linear(decodeLayer1[-1], input_dim1), DispAct())
        self.dec_mean2 = nn.Sequential(nn.Linear(decodeLayer2[-1], input_dim2), MeanAct())
        self.dec_disp2 = nn.Sequential(nn.Linear(decodeLayer2[-1], input_dim2), DispAct())
        self.dec_pi1 = nn.Sequential(nn.Linear(decodeLayer1[-1], input_dim1), nn.Sigmoid())
        self.dec_pi2 = nn.Sequential(nn.Linear(decodeLayer2[-1], input_dim2), nn.Sigmoid())
        self.ffn = se_block(2)
        self.atten = SCSE(2, self.tho)
        self.ffn_norm = nn.LayerNorm(z_dim, eps=1e-6)
        self.nb_loss = NBLoss().to(device)
        self.zinb_loss = ZINBLoss().to(device)
        #self.MSELoss = nn.MSELoss()
        self.device = device

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def aeForward(self, x,x2, g):
        x_c = x + torch.randn_like(x) * self.sigma1
        z1 = self.encoder(g, x_c)
        x_d = x2 + torch.randn_like(x2) * self.sigma1
        z2 = self.encoder2(g, x_d)
        z = torch.stack([z1, z2])
        # z = self.atten3(z)
        r = z.unsqueeze(2).permute([1, 0, 2, 3])  #
        h = r
        r = self.ffn_norm(r)
        r = self.atten(r)
        z = (r+h).permute([1, 0, 2, 3]).squeeze(2)
        z = z[0] + z[1]


        h1 = self.decoder1(z)
        mean1 = self.dec_mean1(h1)
        disp1 = self.dec_disp1(h1)
        pi1 = self.dec_pi1(h1)

        h2 = self.decoder2(z)
        mean2 = self.dec_mean2(h2)
        disp2 = self.dec_disp2(h2)
        pi2 = self.dec_pi2(h2)
        ##
        z3 = self.encoder(g, x)
        z4 = self.encoder2(g, x2)

        z0 = torch.stack([z3, z4])
        # z0 = self.atten3(z0)
        r = z0.unsqueeze(2).permute([1, 0, 2, 3])  #
        h = r
        r = self.ffn_norm(r)
        r = self.atten(r)
        z0 = (r+h).permute([1, 0, 2, 3]).squeeze(2)
        z0 = z0[0] + z0[1]

        return z0, mean1, mean2, disp1, disp2, pi1, pi2

    def aeForward2(self, x, x2, g):
        x_c = x + torch.randn_like(x) * self.sigma1
        z1 = self.encoder(g, x_c)
        x_d = x2 + torch.randn_like(x2) * self.sigma1
        z2 = self.encoder2(g, x_d)
        z = torch.stack([z1, z2])
        # z = self.atten(z)
        r = z.unsqueeze(2).permute([1, 0, 2, 3])  #
        h = r
        r = self.ffn_norm(r)
        r = self.atten(r)
        z = (r+h).permute([1, 0, 2, 3]).squeeze(2)
        z = z[0] + z[1]

        h1 = self.decoder1(z)
        mean1 = self.dec_mean1(h1)
        disp1 = self.dec_disp1(h1)
        pi1 = self.dec_pi1(h1)

        h2 = self.decoder2(z)
        mean2 = self.dec_mean2(h2)
        disp2 = self.dec_disp2(h2)
        pi2 = self.dec_pi2(h2)
        ##
        z3 = self.encoder(g, x)
        z4 = self.encoder2(g, x2)

        z0 = torch.stack([z3, z4])
        # z0 = self.atten3(z0)
        r = z0.unsqueeze(2).permute([1, 0, 2, 3])  #
        h = r
        r = self.ffn_norm(r)
        r = self.atten(r)
        z0 = (r+h).permute([1, 0, 2, 3]).squeeze(2)
        z0 = z0[0] + z0[1]

        q = self.soft_assign(z0)
        return z0, z3, z4, q, mean1, mean2, disp1, disp2, pi1, pi2


    def encodeBatch(self, G_v, X_1, X_2):
        z3 = self.encoder(G_v, X_1)
        z4 = self.encoder2(G_v, X_2)
        z0 = torch.stack([z3, z4])
        # z0 = self.atten3(z0)
        r = z0.unsqueeze(2).permute([1, 0, 2, 3])  #
        h = r
        r = self.ffn_norm(r)
        r = self.atten(r)
        z0 = (r+h).permute([1, 0, 2, 3]).squeeze(2)
        z0 = z0[0] + z0[1]

        return z0, z3, z4

    def train_model(self, X,X2, G,dataloader, lr=0.001, train_iter=400,
                    verbose=True, weights_name=None):

        # A_n = Variable(torch.tensor(A_n, dtype=torch.long))#.to(self.device)
        # size_factor1 = torch.tensor(size_factor1, dtype=torch.float32)#.to(self.device)

        # G = dgl.from_scipy(A_n)
        # G.ndata['feat'] = X
        # X_raw = torch.tensor(X_raw, dtype=torch.float32)

        #X = Variable(X).to(self.device)
        # x1 = Variable(x1).to(self.device)
        # x2 = Variable(x2).to(self.device)
        # G_v = G.to(self.device)
        # A_v = Variable(torch.tensor(A.toarray(), dtype=torch.float32)).to(self.device)
        # X_raw_1 = Variable(X_raw1).to(self.device)
        # X_raw_2 = Variable(X_raw2).to(self.device)
        # size_factor_1 = Variable(size_factor1).to(self.device)
        # size_factor_2 = Variable(size_factor2).to(self.device)
        num = X.shape[0] + X2.shape[0]
        optim_adam = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        print("Training")
        for i in range(train_iter):
            self.train()
            for it, sg in enumerate(dataloader):
                x1 = sg.ndata["x1"]
                x2 = sg.ndata["x2"]
                X_raw1 = sg.ndata['X_raw1']
                X_raw2 = sg.ndata['X_raw2']
                size_factor1 = sg.ndata['size_factor1']
                size_factor2 = sg.ndata['size_factor2']

                latent_z, mean1, mean2, disp1, disp2, pi1, pi2 = self.aeForward(x1,x2, sg)
                loss_zinb1 = self.zinb_loss(x=X_raw1, mean=mean1, disp=disp1, pi=pi1, scale_factor=size_factor1)
                loss_zinb2 = self.zinb_loss(x=X_raw2, mean=mean2, disp=disp2, pi=pi2, scale_factor=size_factor2)
                # sgg = sg.cpu()
                # nx_graph = sgg.to_networkx(node_attrs=None, edge_attrs=None)
                # sgg = nx.adjacency_matrix(nx_graph).toarray()
                # sgg = Variable(torch.tensor(sgg, dtype=torch.float32)).to(self.device)
                # dc = self.dc(latent_z)
                # gcn_loss = F.binary_cross_entropy_with_logits(dc, sgg)
                loss = loss_zinb1 + loss_zinb2  * self.beta
                self.zero_grad()
                loss.backward()
                optim_adam.step()

                if verbose:
                    self.eval()
                    loss_zinb_val = 0
                    _, mean1, mean2, disp1, disp2, pi1, pi2 = self.aeForward(x1,x2, sg)
                    loss_zinb1 = self.zinb_loss(x=X_raw1, mean=mean1, disp=disp1, pi=pi1, scale_factor=size_factor1)
                    loss_zinb2 = self.zinb_loss(x=X_raw2, mean=mean2, disp=disp2, pi=pi2, scale_factor=size_factor2)
                    loss_zinb_val = loss_zinb1.data + loss_zinb2.data  * self.beta
                    print('Iteration:{}, ZINB loss:{:.8f}'.format(i + 1, loss_zinb_val / num))

        torch.save({'ae_state_dict': self.state_dict(),
                    'optimizer_state_dict': optim_adam.state_dict()}, weights_name)

    def target_distribution(self, q):
        p = q ** 2 / q.sum(0)
        return (p.t() / p.sum(1)).t()

    def kldloss(self, p, q):
        c1 = -torch.sum(p * torch.log(q), dim=-1)
        c2 = -torch.sum(p * torch.log(p), dim=-1)
        return torch.mean(c1 - c2)

    def cal_latent(self, z):
        sum_y = torch.sum(torch.square(z), dim=1)
        num = -2.0 * torch.matmul(z, z.t()) + torch.reshape(sum_y, [-1, 1]) + sum_y
        num = num / self.alpha
        num = torch.pow(1.0 + num, -(self.alpha + 1.0) / 2.0)
        zerodiag_num = num - torch.diag(torch.diag(num))
        latent_p = (zerodiag_num.t() / torch.sum(zerodiag_num, dim=1)).t()
        return num, latent_p

    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q

    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=-1))
        kldloss = kld(p, q)
        return kldloss

    def kmeans_loss(self, z):
        dist1 = self.tau * torch.sum(torch.square(z.unsqueeze(1) - self.mu.cuda()), dim=2)  #
        temp_dist1 = dist1 - torch.reshape(torch.mean(dist1, dim=1), [-1, 1])
        q = torch.exp(-temp_dist1)
        q = (q.t() / torch.sum(q, dim=1)).t()
        q = torch.pow(q, 2)
        q = (q.t() / torch.sum(q, dim=1)).t()
        dist2 = dist1 * q
        return dist1, torch.mean(torch.sum(dist2, dim=1))

    # def cosine(self, x):
    #     #cosine =np.corrcoef(x.data.cpu().numpy())
    #     cosine = torch.pow(torch.sum(x ** 2.0, dim=1), 0.5)
    #     cosine = (x.t() / cosine).t()
    #     cosine = torch.mm(cosine, cosine.t())
    #     return cosine
    #
    # def cosine_loss(self, z1, z2):
    #     cosine1 = self.cosine(z1)
    #     cosine2 = self.cosine(z2)
    #     closs = self.MSELoss(cosine1, cosine2)
    #     return closs

    def fit(self, X,X2, G,dataloader, n_clusters,
            y=None, lr=0.001, batch_size=256, num_epochs=1, update_interval=1, tol=1e-3):
        X_1 = Variable(X).to(self.device)
        X_2 = Variable(X2).to(self.device)
        G_v = G.to(self.device)

        self.mu = Parameter(torch.Tensor(n_clusters, self.z_dim).to(self.device))

        print("Initializing cluster centers with kmeans")
        kmeans = KMeans(n_clusters, n_init=20)
        Zdata,_,_ = self.encodeBatch(G_v, X_1, X_2)
        Zdata = Zdata.data.cpu().numpy()
        #Zdata = self.encoder(G_v, X_v).data.cpu().numpy()
        self.y_pred = kmeans.fit_predict(Zdata)
        self.y_pred_last = self.y_pred
        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))

        final_pred, final_Zdata = self.y_pred, Zdata
        if y is not None:
            acc, nmi, ari = eval_cluster(y, self.y_pred)
            print('Initializing kmeans: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))
            # print('Initializing kmeans KNN ACC= %.4f' % knn_ACC(p_, self.y_pred))

        num = X.shape[0]
        final_acc, final_nmi, final_ari, final_epoch = 0, 0, 0, 0
        optim_adam = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95)
        # optim_adam = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        for epoch in range(num_epochs):
            self.eval()
            with torch.no_grad():
                if epoch % update_interval == 0:
                    # update the targe distribution p
                    Zdata, zr, zp = self.encodeBatch(G_v, X_1, X_2)
                    q = self.soft_assign(Zdata)
                    self.y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
                    # q1 = self.soft_assign(zr)
                    # self.y_pred1 = torch.argmax(q1, dim=1).data.cpu().numpy()
                    # q2 = self.soft_assign(zp)
                    # self.y_pred2 = torch.argmax(q2, dim=1).data.cpu().numpy()


                if y is not None:
                    acc, nmi, ari = eval_cluster(y, self.y_pred)


            delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / num
            self.y_pred_last = self.y_pred
            if final_ari < ari:
                final_ari = ari
                final_pred = self.y_pred
                final_Zdata = Zdata
            if epoch > 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print("Reach tolerance threshold. Stopping training.")
                break
            total_loss=0
            kl_val = 0
            zinb_val = 0
            # update clustering
            self.train()
            for it, sg in enumerate(dataloader):
                x1 = sg.ndata["x1"]
                x2 = sg.ndata["x2"]
                X_raw1 = sg.ndata['X_raw1']
                X_raw2 = sg.ndata['X_raw2']
                size_factor1 = sg.ndata['size_factor1']
                size_factor2 = sg.ndata['size_factor2']

                z, z3, z4, qbatch, mean1, mean2, disp1, disp2, pi1, pi2 = self.aeForward2(x1, x2, sg)
                loss_zinb1 = self.zinb_loss(x=X_raw1, mean=mean1, disp=disp1, pi=pi1, scale_factor=size_factor1)
                loss_zinb2 = self.zinb_loss(x=X_raw2, mean=mean2, disp=disp2, pi=pi2, scale_factor=size_factor2)
                # nu, lq1 = self.cal_latent(z)
                # target1 = self.target_distribution(lq1)
                # lq1 = lq1 + torch.diag(torch.diag(nu))
                # target1 = target1 + torch.diag(torch.diag(nu))
                # kl_loss = self.kldloss(target1, lq1)
                p = self.target_distribution(qbatch).data
                target = Variable(p).to(self.device)
                loss_cluster = self.cluster_loss(target, qbatch)
                #loss_constrast = self.cosine_loss(z3, z4)

                # sgg = sg.cpu()
                # nx_graph = sgg.to_networkx(node_attrs=None, edge_attrs=None)
                # sgg = nx.adjacency_matrix(nx_graph).toarray()
                # sgg = Variable(torch.tensor(sgg, dtype=torch.float32)).to(self.device)
                # dc = self.dc(z)
                # gcn_loss = F.binary_cross_entropy_with_logits(dc, sgg)
                loss = loss_zinb1 + loss_zinb2  * self.beta + self.gamma * loss_cluster #+ loss_constrast * 0.001
                optim_adam.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.mu, 1)
                optim_adam.step()
                total_loss += (loss_zinb1 + loss_zinb2 * self.beta).data / num + loss_cluster.data / num #+ loss_constrast.data / num
                kl_val += loss_cluster
                zinb_val += loss_zinb1+ loss_zinb2 * self.beta


            print(
                'Clustering Iteration:{}, Total loss:{:.8f}, ZINB loss1:{:.8f}, Cluster loss:{:.8f}'.format(
                    epoch + 1,
                    total_loss, zinb_val.data / num,  kl_val.data / num))

        return final_pred, final_Zdata, total_loss, epoch + 1, zr, zp
