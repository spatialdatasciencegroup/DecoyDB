
# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from utils import AverageMeter
from SchNet import SchNet
from predataset_SchNet import PreGraphDatasetD,PreGraphDatasetV
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.nn import DataParallel, global_max_pool,global_mean_pool, GINConv
from config.config_dict import Config
from log.train_logger import TrainLogger
import numpy as np
from utils import *
from sklearn.metrics import mean_squared_error
import time
from infoloss import InfoNCE, InfoNCESep
print("Available GPUs:", torch.cuda.device_count())
from matplotlib import pyplot as plt

def val(model,model1, dataloader, device):
    model.eval()
    criterion2 = nn.MSELoss()
    criterion = InfoNCE(temperature=0.1,batch_size = 16,n_views=11)
    criterion1 = InfoNCESep(temperature=0.1,batch_size = 16,n_views=11)
    pred_list = []
    label_list = []
    for data in dataloader:
        with torch.no_grad():
            InputBatch1 = []
            InputBatch2 = []
            for b in data:
                if len(b)!=0:
                    InputBatch1 +=b[:11]
                    InputBatch2 +=b[11:]
                else:
                    InputBatch1 +=b
                    InputBatch2 +=b
            InputBatch1 = [data.cuda() for data in InputBatch1]
            InputBatch2 = [data.cuda() for data in InputBatch2]
            for item in InputBatch2:
                item.pos.requires_grad_()
            pred,label,x1 = model(InputBatch1)
            pred1,label1,x11 = model1(InputBatch1)
            x1 = x1.view(-1,11,128)
            x11 = x11.view(-1,11,128)
            loss = criterion(x1, label)
            loss1 = criterion1(x1,x11,label)
            loss =0.5*loss+loss1
            pred2,label2,x12 = model(InputBatch2)

            label_list.append(loss)

    rmse = sum(label_list)/len(label_list)
    model.train()

    return rmse, 1
def denoising_loss(y,data,u,std,batch_size,cri):
    loss = []
    for i in range(len(data)//10):
        for j in range(10):
            d = data[i*10+j].pos
            f = torch.autograd.grad(y[i*10+j], d, create_graph=True, retain_graph=True)[0]
            u1 = u[i*10+j]
            
            target = (u1/std)*torch.ones_like(d).cuda()
            loss.append(cri(f,target).view(-1))
    return torch.cat(loss).mean()
# %%
if __name__ == '__main__':
    cfg = 'TrainConfig_SchNet'
    
    config = Config(cfg)
    args = config.get_config()
    graph_type = args.get("graph_type")
    save_model = args.get("save_model")
    batch_size = 8
    data_root = args.get('data_root')
    epochs = args.get('epochs')
    epochs = 20
    repeats = args.get('repeat')
    early_stop_epoch = args.get("early_stop_epoch")
    repeat = 5
    train_list = []
    val_list = []
    for repeat in range(repeats):
        args['repeat'] = repeat

        train_dir = os.path.join(data_root, 'train')
        valid_dir = os.path.join(data_root, 'valid')
        test2013_dir = os.path.join(data_root, 'test2013')
        test2016_dir = os.path.join(data_root, 'test2016')
        test2019_dir = os.path.join(data_root, 'test2019')

        train_df = pd.read_csv(os.path.join(data_root, 'train.csv'))
        valid_df = pd.read_csv(os.path.join(data_root, 'valid.csv'))
        test2013_df = pd.read_csv(os.path.join(data_root, 'test2013.csv'))
        test2016_df = pd.read_csv(os.path.join(data_root, 'test2016.csv'))
        test2019_df = pd.read_csv(os.path.join(data_root, 'test2019.csv'))


        train_set = PreGraphDatasetD("./data/pretrain", train_df, graph_type=graph_type, create=False)
        valid_set = PreGraphDatasetD("./data/pretrain", valid_df, graph_type=graph_type, create=False)
        
        train_set.data_name = train_set.data_name[0:10000]
        valid_set.data_name = train_set.data_name[-1000:]
        train_set.load()
        valid_set.load()
        test2013_set = PreGraphDatasetV(test2013_dir, test2013_df, graph_type=graph_type, create=False)
        test2016_set = PreGraphDatasetV(test2016_dir, test2016_df, graph_type=graph_type, create=False)
        test2019_set = PreGraphDatasetV(test2019_dir, test2019_df, graph_type=graph_type, create=False)
        

        
        train_loader = DataListLoader(train_set,batch_size = batch_size,shuffle = True)
        
        valid_loader = DataListLoader(valid_set, batch_size=batch_size, shuffle=False)
        test2016_loader = DataListLoader(test2016_set, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True,timeout = 2000)
        test2013_loader = DataListLoader(test2013_set, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True,timeout = 2000)
        test2019_loader = DataListLoader(test2019_set, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True,timeout = 2000)

        logger = TrainLogger(args,'preinfonce', cfg, create=True)
        logger.info(__file__)
        logger.info(f"train data: {len(train_set)}")
        logger.info(f"valid data: {len(valid_set)}")
        logger.info(f"test2013 data: {len(test2013_set)}")
        logger.info(f"test2016 data: {len(test2016_set)}")
        logger.info(f"test2019 data: {len(test2019_set)}")
        with open(os.path.join(logger.get_model_dir(),f'repeats_{repeat}.pkl'), 'wb') as file1:
            pickle.dump(train_set,file1)
        model =  SchNet(num_interactions=3).cuda()

        model = DataParallel(model)

        
        optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
        criterion2 = nn.MSELoss()
        criterion = InfoNCE(temperature=0.1,batch_size = 16,n_views=11,alpha = 0.8)
        criterion1 = InfoNCESep(temperature=0.1,batch_size = 16,n_views=11)
        
        running_loss = AverageMeter()
        running_acc = AverageMeter()
        running_best_mse = BestMeter("min")
        best_model_list = []
        
        model.train()
        for epoch in range(epochs):
            for data in train_loader:
                InputBatch1 = []
                InputBatch2 = []
                for b in data:
                    if len(b)!=0:
                        InputBatch1 +=b[:11]
                        InputBatch2 +=b[11:]
                    else:
                        InputBatch1 +=b
                        InputBatch2 +=b
                InputBatch1 = [data.cuda() for data in InputBatch1]
                InputBatch2 = [data.cuda() for data in InputBatch2]
                for item in InputBatch2:
                    item.pos.requires_grad_()
                pred,label,x1 = model(InputBatch1)
                # label = data.y
                x1 = x1.view(-1,11,128)
                loss = criterion(x1, label)

                pred2,label2,x12 = model(InputBatch2)
                loss2 = 0.1*denoising_loss(pred2,InputBatch2,label2,2,batch_size,criterion2)
                loss = loss + loss2
                # break
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss.update(loss.item(), label.size(0)) 
                
                

            epoch_loss = running_loss.get_average()
            epoch_rmse = np.sqrt(epoch_loss)
            running_loss.reset()
            valid_rmse, valid_pr = val(model,model1, valid_loader, 'cuda0')
            msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse-%.4f, valid_pr-%.4f" \
                    % (epoch, epoch_loss, epoch_rmse, valid_rmse, valid_pr)
            train_list.append(epoch_rmse.item())
            val_list.append(valid_rmse.item())
            logger.info(msg)

            if valid_rmse < running_best_mse.get_best():
                running_best_mse.update(valid_rmse)
                if save_model:
                    msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse-%.4f, valid_pr-%.4f" \
                    % (epoch, epoch_loss, epoch_rmse, valid_rmse, valid_pr)
                    model_path = os.path.join(logger.get_model_dir(), msg + '.pt')
                    best_model_list.append(model_path)
                    save_model_dict(model, logger.get_model_dir(), msg)
            else:
                count = running_best_mse.counter()
                if count > early_stop_epoch:
                    best_mse = running_best_mse.get_best()
                    msg = "best_rmse: %.4f" % best_mse
                    logger.info(f"early stop in epoch {epoch}")
                    logger.info(msg)
                    break_flag = True
                    break

        # final testing
        load_model_dict(model, best_model_list[-1])
        device = 'cuda'
        valid_rmse, valid_pr = val(model,model1, valid_loader, device)

        save_path = os.path.join(logger.get_model_dir(), "loss.pdf")
        plot_training_curve(line_1_x=range(len(train_list)),
						line_1_y=train_list,
						line_2_x=range(len(val_list)),
						line_2_y=val_list,
						save_path=save_path,
						y_label="RMSE")
