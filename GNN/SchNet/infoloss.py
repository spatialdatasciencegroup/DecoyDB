import torch
import torch.nn as nn
import torch.nn.functional as F
def gaussian(x, a=1, b=0, c=10):
    return a * torch.exp(-(x - b)**2 / (c**2))
def reweight(x,alpha = 1.5):
    x[x<2] = 0

    return (alpha*(x/20))

def zeros(x):
    x[x==0]=0
    x[x!=0] = 1
    return x
class InfoNCE(nn.Module):
    def __init__(self, temperature=0.1,batch_size = 64,n_views=5,alpha=1.5):
        super(InfoNCE, self).__init__()
        self.temperature = temperature
        self.n_views = n_views
        self.batch_size = batch_size
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.alpha = alpha
    def info_nce_loss(self, features,y):

            anchor_label = torch.arange(features.shape[0])
            labels = torch.cat([torch.arange(features.shape[0]) for i in range(self.n_views)], dim=0)
            labels = (anchor_label.unsqueeze(1)==labels.unsqueeze(0)).float()
            labels = labels.cuda()

            weights = torch.ones_like(labels).cuda()*10
            ones_indices = (labels == 1).nonzero(as_tuple=True)
            weights[ones_indices] = y

            
            zero = zeros(weights)
            weights = reweight(weights,self.alpha)

            features = F.normalize(features, dim=1)
            anchor = features[:,0,:]
            features = features.permute(1,0,2)
            features = features.reshape(-1,features.shape[-1])
            anchor = anchor.reshape(-1,features.shape[-1])
            similarity_matrix = torch.matmul(anchor, features.T)

            similarity_matrix_reweight = weights*similarity_matrix
            labels_r = labels.detach().clone()
            labels_r[:features.shape[0],:features.shape[0]]=0

            positives = similarity_matrix[labels_r.bool()].view(labels.shape[0], -1)
    
            # select only the negatives the negatives
            
            negatives = similarity_matrix_reweight[~labels.bool()].view(similarity_matrix_reweight.shape[0], -1)
            logits = torch.cat([positives, negatives], dim=1)
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    
            logits = logits / self.temperature
            return logits, labels
    def forward(self, features,y):
        logits, label = self.info_nce_loss(features,y)
        # print(logits.shape)
        loss = self.criterion(logits, label)
        return loss
class InfoNCESep(nn.Module):
    def __init__(self, temperature=0.1,batch_size = 64,n_views=5):
        super(InfoNCESep, self).__init__()
        self.temperature = temperature
        self.n_views = n_views
        self.batch_size = batch_size
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
    def info_nce_loss(self, features,features1,y):

            anchor_label = torch.cat([torch.arange(features.shape[0]) ], dim=0)
            labels = torch.cat([torch.arange(features.shape[0])], dim=0)
            labels = (anchor_label.unsqueeze(1)==labels.unsqueeze(0)).float()
            labels = labels.cuda()

            weights = torch.ones_like(labels).cuda()*10
            ones_indices = (labels == 1).nonzero(as_tuple=True)

            features = F.normalize(features[:,0,:], dim=1)
            anchor = features

            features1 = features1[:,0,:]
            features1 = features1.reshape(-1,features1.shape[-1])
            
            anchor = anchor.reshape(-1,features1.shape[-1])

            similarity_matrix = torch.matmul(anchor, features1.T)

            similarity_matrix_reweight = similarity_matrix

            labels_r = labels

            positives = similarity_matrix[labels_r.bool()].view(labels.shape[0], -1)
    
            # select only the negatives the negatives
            
            negatives = similarity_matrix_reweight[~labels.bool()].view(similarity_matrix_reweight.shape[0], -1)
            logits = torch.cat([positives, negatives], dim=1)
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    
            logits = logits / self.temperature
            return logits, labels
    def forward(self, features,features1,y):
        logits, label = self.info_nce_loss(features,features1,y)
        loss = self.criterion(logits, label)
        return loss