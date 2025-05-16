
# # %%
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from utils import AverageMeter
from GIGN import GIGN
from predataset_GIGN import PreGraphDatasetD,PreGraphDatasetV
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import DataParallel, global_max_pool,global_mean_pool, GINConv
from config.config_dict import Config
from log.train_logger import TrainLogger
import numpy as np
from utils import *
from sklearn.metrics import mean_squared_error
from infoloss import InfoNCE,InfoNCESep
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.optim.lr_scheduler import SequentialLR
torch.cuda.empty_cache()
# %%
def val(model,model1, dataloader, device):
    model.eval()
    criterion2 = nn.MSELoss()
    criterion = InfoNCE(temperature=0.1,batch_size = 16,n_views=11)
    criterion1 = InfoNCESep(temperature=0.1,batch_size = 16,n_views=11)
    pred_list = []
    label_list = []
    for data in dataloader:
        # data = data.to(device)
        with torch.no_grad():
            InputBatch1 = []
            InputBatch2 = []
            for b in data:
                if len(b)!=0:
                    InputBatch1 +=b[:11]
                    InputBatch2 +=b[11:14]
                else:
                    InputBatch1 +=b
                    InputBatch2 +=b
            InputBatch1 = [data.cuda() for data in InputBatch1]
            InputBatch2 = [data.cuda() for data in InputBatch2]

            pred,label,x1 = model(InputBatch1)
            pred1,label1,x11 = model1(InputBatch1)

            x1 = x1.view(-1,11,256)

            loss = criterion(x1, label)

            loss = loss 

            label_list.append(loss)

            

    rmse = sum(label_list)/len(label_list)
    model.train()

    return rmse, 1
def denoising_loss(y,data,u,std,batch_size,cri):
    loss = []
    num_sample = 3
    # print(len(data))
    for i in range(len(data)//num_sample):
        for j in range(num_sample):
            d = data[i*num_sample+j].pos

            f = torch.autograd.grad(y[i*num_sample+j], d, create_graph=True, retain_graph=True)[0]

            u1 = data[i*num_sample+j].y
            target = u1

            loss.append(cri(f,target).view(-1))

    return torch.cat(loss).mean()
                
import psutil

# %%
if __name__ == '__main__':
    cfg = 'TrainConfig_GIGN'

    print("Available GPUs:", torch.cuda.device_count())
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
    repeats = 1
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
        valid_set.data_name = train_set.data_name[-1000:]
        train_set.data_name = train_set.data_name[:-1000]
        
        pdbids_upper_train = train_df['pdbid'].str.upper().tolist()
        pdbids_upper_valid = valid_df['pdbid'].str.upper().tolist()
        pdbids_upper_test = test2016_df['pdbid'].str.upper().tolist()
        pdbids_upper = pdbids_upper_train + pdbids_upper_valid + pdbids_upper_test
        valid_set.data_name = [item for item in valid_set.data_name if item.split('-')[0] not in pdbids_upper]
        train_set.data_name  = [item for item in train_set.data_name if item.split('-')[0] not in pdbids_upper]

        train_set.load()
        valid_set.load()
       
        
        
        test2013_set = PreGraphDatasetV(test2013_dir, test2013_df, graph_type=graph_type, create=False)
        test2016_set = PreGraphDatasetV(test2016_dir, test2016_df, graph_type=graph_type, create=False)
        test2019_set = PreGraphDatasetV(test2019_dir, test2019_df, graph_type=graph_type, create=False)

        train_loader = DataListLoader(train_set,batch_size = batch_size,shuffle = True)
        valid_loader = DataListLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4)
        test2016_loader = DataListLoader(test2016_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test2013_loader = DataListLoader(test2013_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test2019_loader = DataListLoader(test2019_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        logger = TrainLogger(args,'pretrain_de', cfg, create=True)
        logger.info(__file__)
        logger.info(f"train data: {len(train_set)}")
        logger.info(f"valid data: {len(valid_set)}")
        logger.info(f"test2013 data: {len(test2013_set)}")
        logger.info(f"test2016 data: {len(test2016_set)}")
        logger.info(f"test2019 data: {len(test2019_set)}")
        with open(os.path.join(logger.get_model_dir(),f'repeats_{repeat}.pkl'), 'wb') as file1:
            pickle.dump([train_set.data_name,valid_set.data_name],file1)

        model = GIGN(35, 256,15).cuda()

        model = DataParallel(model)
        for name, param in model.named_parameters():
            if ('output' in name ):
                # param.requires_grad = False
                # print(param.shape)
                continue
            else:
                if len(param.shape)>1:
                    nn.init.kaiming_normal_(param)
        model1 = GIGN(35, 256,15).cuda()
        model1 = DataParallel(model1)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        criterion2 = nn.MSELoss()
        criterion = InfoNCE(temperature=0.1,batch_size = 16,n_views=11,alpha = 1.5)
        criterion1 = InfoNCESep(temperature=0.1,batch_size = 16,n_views=11)
        running_loss = AverageMeter()
        running_acc = AverageMeter()
        running_best_mse = BestMeter("min")
        best_model_list = []
        device = 'cuda0'
        model.train()
        print(len(train_set))
        print(len(train_set.datalist))
        iteration = 0
        for epoch in range(epochs):
            
            for data in train_loader:
                iteration+=1
                process = psutil.Process()
                InputBatch1 = []
                InputBatch2 = []
                for b in data:
                    if len(b)!=0:
                        InputBatch1 +=b[:11]
                        InputBatch2 +=b[11:14]
                    else:
                        InputBatch1 +=b
                        InputBatch2 +=b

                InputBatch1 = [data.cuda() for data in InputBatch1]
                InputBatch2 = [data.cuda() for data in InputBatch2]
                for item in InputBatch2:
                    item.pos.requires_grad_()
                pred,label,x1 = model(InputBatch1)
                

                x1 = x1.view(-1,11,256)

                loss = criterion(x1, label)

                pred2,label2,x12 = model(InputBatch2)

                loss2 = denoising_loss(pred2,InputBatch2,label2,2,batch_size,criterion2)
                loss = loss + 0.2*loss2
                # break
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss.update(loss.item(), label2.size(0))

            epoch_loss = running_loss.get_average()
            epoch_rmse = np.sqrt(epoch_loss)
            running_loss.reset()

            # start validating
            valid_rmse, valid_pr = val(model,model1, valid_loader, 'cuda0')
            msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse-%.4f, valid_pr-%.4f" \
                    % (epoch, epoch_loss, epoch_rmse, valid_rmse, valid_pr)
            train_list.append(epoch_rmse.item())
            val_list.append(valid_rmse.item())
            logger.info(msg)
            
            model_path = os.path.join(logger.get_model_dir(), msg + '.pt')
            best_model_list.append(model_path)
            save_model_dict(model, logger.get_model_dir(), msg)
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
        device = 'cuda0'
        valid_rmse, valid_pr = val(model,model1, valid_loader,'cuda0')

        test2013_rmse, test2013_pr = [0,0]
        test2016_rmse, test2016_pr = [0,0]
        test2019_rmse, test2019_pr = [0,0]
        save_path = os.path.join(logger.get_model_dir(), "loss.pdf")
        plot_training_curve(line_1_x=range(len(train_list)),
						line_1_y=train_list,
						line_2_x=range(len(val_list)),
						line_2_y=val_list,
						save_path=save_path,
						y_label="RMSE")
        msg = "valid_rmse-%.4f, valid_pr-%.4f, test2013_rmse-%.4f, test2013_pr-%.4f, test2016_rmse-%.4f, test2016_pr-%.4f, test2019_rmse-%.4f, test2019_pr-%.4f," \
                    % (valid_rmse, valid_pr, test2013_rmse, test2013_pr, test2016_rmse, test2016_pr, test2019_rmse, test2019_pr)

        logger.info(msg)
        
        
