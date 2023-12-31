from .rgcn_model import RGCN
from dgl import mean_nodes
import torch.nn as nn
import torch
import torch.nn.functional as F
"""
File based off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""


class GraphClassifier(nn.Module):
    def __init__(self, params, relation2id):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        self.relation2id = relation2id
        self.dropout = nn.Dropout(p = 0.3)
        self.relu = nn.ReLU()
        #print(len(relation2id))
        self.train_rels = params.train_rels
        self.relations = params.num_rels
        self.gnn = RGCN(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)
        #self.rel_emb = nn.Embedding(self.params.num_rels, self.params.rel_emb_dim, sparse=False)
        self.mp_layer1 = nn.Linear(self.params.feat_dim, self.params.emb_dim)
        self.mp_layer2 = nn.Linear(self.params.emb_dim, self.params.emb_dim)

        self.mp_layer1_p = nn.Linear(self.params.pfeat_dim, self.params.emb_dim)
        self.mp_layer2_p = nn.Linear(self.params.emb_dim, self.params.emb_dim)
        
        if self.params.add_ht_emb and self.params.add_sb_emb:
            if self.params.add_feat_emb and self.params.add_transe_emb:
                self.fc_layer = nn.Linear(3 * (1+self.params.num_gcn_layers) * self.params.emb_dim + 2*self.params.emb_dim, self.train_rels)
            elif self.params.add_feat_emb :
                self.fc_layer = nn.Linear(3 * (self.params.num_gcn_layers) * self.params.emb_dim + 2*self.params.emb_dim, self.train_rels)
            else:
                self.fc_layer = nn.Linear(3 * (1+self.params.num_gcn_layers) * self.params.emb_dim, self.train_rels)
        elif self.params.add_ht_emb:
            self.fc_layer = nn.Linear(2 * (1+self.params.num_gcn_layers) * self.params.emb_dim, self.train_rels)
        else:
            self.fc_layer = nn.Linear(self.params.num_gcn_layers * self.params.emb_dim, self.train_rels)
        #print(self.fc_layer)
    def drug_feat(self, emb):
        self.drugfeat = emb
    def pro_feat(self, emb):
        self.profeat = emb
    def pro_ind(self, pind): # profeat 참조 인덱스
        self.proind = pind 
    def drug_ind(self, dind):
        self.drugind = dind

    def forward(self, data):
        g = data
        # print(type(g)) # dgl.graph.DGLGraph
        # print(g) # 그래프 요약
        g.ndata['h'] = self.gnn(g) # Get feature dictionary of all nodes
        # print(len(g.ndata['h']))  # node 개수
        # print(data)
        # print('repr:',g.ndata['repr'], g.ndata['repr'].shape)
        g_out = mean_nodes(g, 'repr')
        #print('g_out', g_out.shape)
        # print(g_out.shape,g.ndata['h'].shape)


        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['repr'][head_ids]

        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['repr'][tail_ids]
#        print(g.ndata['idx'][head_ids], g.ndata['idx'][tail_ids],  g.ndata['idx'][tail_ids].shape)
        #print(g.ndata['idx'][head_ids])
        # print(self.proind[g.ndata['idx'][head_ids]])
        head_feat = self.profeat[self.proind[g.ndata['idx'][head_ids]]] # 교수님이 profeat 추가한부분
 #       head_feat = self.profeat[g.ndata['idx'][head_ids]] # 교수님이 profeat 추가한브븐
        tail_feat = self.drugfeat[self.drugind[g.ndata['idx'][tail_ids]]]
        #print(head_feat.shape, tail_feat.shape)
        # drug_feat = self.drugfeat[drug_idx]
        # print(drug_feat, drug_feat.shape)
        if self.params.add_feat_emb:
            fuse_feat1 = self.mp_layer2_p( self.relu( self.dropout( self.mp_layer1_p(
                            head_feat #torch.cat([head_feat, tail_feat], dim = 1)
                        ))))
            fuse_feat2 = self.mp_layer2( self.relu( self.dropout( self.mp_layer1(
                            tail_feat #torch.cat([head_feat, tail_feat], dim = 1)
                        ))))
            fuse_feat = torch.cat([fuse_feat1, fuse_feat2], dim = 1)
        if self.params.add_ht_emb and self.params.add_sb_emb:
            if self.params.add_feat_emb and self.params.add_transe_emb:
                # print(g_out.shape, head_embs.shape, tail_embs.shape, fuse_feat.shape)
                g_rep = torch.cat([g_out.view(-1, (1+self.params.num_gcn_layers) * self.params.emb_dim),
                                   head_embs.view(-1, (1+self.params.num_gcn_layers) * self.params.emb_dim),
                                   tail_embs.view(-1, (1+self.params.num_gcn_layers) * self.params.emb_dim),
                                   fuse_feat.view(-1, 2*self.params.emb_dim)
                                   ], dim=1)
            elif self.params.add_feat_emb:
                g_rep = torch.cat([g_out.view(-1, (self.params.num_gcn_layers) * self.params.emb_dim),
                                   head_embs.view(-1, (self.params.num_gcn_layers) * self.params.emb_dim),
                                   tail_embs.view(-1, (self.params.num_gcn_layers) * self.params.emb_dim),
                                   fuse_feat.view(-1, 2*self.params.emb_dim)
                                   ], dim=1)
            else:
                g_rep = torch.cat([g_out.view(-1, (1+self.params.num_gcn_layers) * self.params.emb_dim),
                                   head_embs.view(-1, (1+self.params.num_gcn_layers) * self.params.emb_dim),
                                   tail_embs.view(-1, (1+self.params.num_gcn_layers) * self.params.emb_dim),
                                   #fuse_feat.view(-1, 2*self.params.emb_dim)
                                   ], dim=1)

        elif self.params.add_ht_emb:
            g_rep = torch.cat([
                                head_embs.view(-1, (1+self.params.num_gcn_layers) * self.params.emb_dim),
                                tail_embs.view(-1, (1+self.params.num_gcn_layers) * self.params.emb_dim)
                               ], dim=1)
        else:
            g_rep = g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim)
        # print(g_rep.shape) # torch.Size([{batch_size}, 128])
        # exit(0)
        #print(g_rep.shape, self.params.add_ht_emb, self.params.add_sb_emb)
        output = self.fc_layer(F.dropout(g_rep, p =0.3))

        # print(output.shape) # torch.Size([{batch_size}, {number of relations}])
        # exit(0)
        # print(head_ids.detach().cpu().numpy(), tail_ids.detach().cpu().numpy())
        return output
