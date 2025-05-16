
# %%
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "4"
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from utils import AverageMeter
from SchNet import SchNet
from torch_geometric.nn import DataParallel
from dataset_SchNet import GraphDataset, PLIDataLoader
from config.config_dict import Config
from log.train_logger import TrainLogger
import numpy as np
from utils import *
from sklearn.metrics import mean_squared_error
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("pretrain", help="if pretrain")
parser.add_argument("sample_num", help="the number of samples for training")
# %%
def val(model, dataloader, device):
    model.eval()

    pred_list = []
    label_list = []
    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            # x, edge_index_intra, edge_index_inter, pos, batch = \
            #         data.x, data.edge_index_intra, data.edge_index_inter, data.pos, data.batch
            # edge_index = torch.cat([edge_index_intra, edge_index_inter], dim=-1)
            pred,y,x1 = model(data)
            label = data.y

            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            
    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    coff = np.corrcoef(pred, label)[0, 1]
    rmse = np.sqrt(mean_squared_error(label, pred))

    model.train()

    return rmse, coff

# %%
if __name__ == '__main__':
    cfg = 'TrainConfig_SchNet'
    config = Config(cfg)
    args = config.get_config()
    args1 = parser.parse_args()
    graph_type = args.get("graph_type")
    save_model = args.get("save_model")
    batch_size = args.get("batch_size")
    data_root = args.get('data_root')
    epochs = args.get('epochs')
    epochs = 300
    repeats = args.get('repeat')
    early_stop_epoch = args.get("early_stop_epoch")
    early_stop_epoch = 30
    repeats = 7
    torch.cuda.manual_seed(2024)
    torch.cuda.manual_seed_all(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)
    random.seed(2024)
    sample_num = int(args1.sample_num)
    print(sample_num)
    pretrain = int(args1.pretrain)
    print(pretrain)
    test2013_list = []
    test2016_list = []
    test2019_list = []
    test2013_p = []
    test2016_p = []
    test2019_p = []
    for repeat in range(repeats):
        args['repeat'] = repeat

        train_dir = os.path.join(data_root, 'train2013')
        valid_dir = os.path.join(data_root, 'valid')
        test2013_dir = os.path.join(data_root, 'test2013')
        test2016_dir = os.path.join(data_root, 'test2016')
        test2019_dir = os.path.join(data_root, 'test2019')
        train_df = pd.read_csv(os.path.join('./data', 'train2013.csv'))

        valid_df = pd.read_csv(os.path.join(data_root, 'valid.csv'))
        test2013_df = pd.read_csv(os.path.join(data_root, 'test2013.csv'))
        test2016_df = pd.read_csv(os.path.join(data_root, 'test2016.csv'))
        test2019_df = pd.read_csv(os.path.join(data_root, 'test2019.csv'))

        train_set = GraphDataset(train_dir, train_df, graph_type=graph_type, create=False)
        valid_set = GraphDataset(valid_dir, valid_df, graph_type=graph_type, create=False)
        test2013_set = GraphDataset(test2013_dir, test2013_df, graph_type=graph_type, create=False)
        test2016_set = GraphDataset(test2016_dir, test2016_df, graph_type=graph_type, create=False)
        test2019_set = GraphDataset(test2019_dir, test2019_df, graph_type=graph_type, create=False)
        with open(os.path.join(f'./data/data_samples_2013/',f'repeats_{repeat}.pkl'), 'rb') as file1:
            all_path = pickle.load(file1)
        for i in range(len(all_path)):
            name_list = all_path[i].split('/')
            name1 = name_list[-1].split('-')
            name1 = 'Graph_SchNet-' + name1[-1]

            name_list[-1] = name1
            name_recover = ''
            for j in name_list:
                name_recover = name_recover + j + '/'
            name_recover = name_recover[:-1]
            all_path[i] = name_recover
        
        train_set.graph_paths = all_path[:sample_num]
        valid_set.graph_paths = all_path[-1000:]
        print(len(train_set.graph_paths))
        print(len(train_set))
        train_loader = PLIDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
        valid_loader = PLIDataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=1)
        test2016_loader = PLIDataLoader(test2016_set, batch_size=batch_size, shuffle=False, num_workers=1)
        test2013_loader = PLIDataLoader(test2013_set, batch_size=batch_size, shuffle=False, num_workers=1)
        test2019_loader = PLIDataLoader(test2019_set, batch_size=batch_size, shuffle=False, num_workers=1)
        device = torch.device('cuda:0')
        model = SchNet(num_interactions=3).to(device)
        if pretrain==1:
            model = DataParallel(model)
            model.load_state_dict(torch.load("./model/20240925_024226_SchNet_repeat0_preinfonce/model/epoch-9, train_loss-0.0402, train_rmse-0.2005, valid_rmse-2755.5054, valid_pr-1.0000.pt"))
            model = model.module
            logger = TrainLogger(args,'finetuning_PT', cfg, create=True)
        else:
            logger = TrainLogger(args,'finetuning_noPT', cfg, create=True)
        logger.info(__file__)
        logger.info(f"train data: {len(train_set)}")
        logger.info(f"valid data: {len(valid_set)}")
        logger.info(f"test2013 data: {len(test2013_set)}")
        logger.info(f"test2016 data: {len(test2016_set)}")
        logger.info(f"test2019 data: {len(test2019_set)}")

        
        optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience =8,factor=0.1,min_lr = 0.00001)
        criterion = nn.MSELoss()

        running_loss = AverageMeter()
        running_acc = AverageMeter()
        running_best_mse = BestMeter("min")
        best_model_list = []
        
        model.train()
        for epoch in range(epochs):
            for data in train_loader:
                data = data.to(device)
                z, pos, batch = data.x, data.pos, data.batch
                pred,y,x1 = model(data)
                label = data.y

                loss = criterion(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss.update(loss.item(), label.size(0)) 

            epoch_loss = running_loss.get_average()
            epoch_rmse = np.sqrt(epoch_loss)
            running_loss.reset()

            # start validating
            valid_rmse, valid_pr = val(model, valid_loader, device)
            msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse-%.4f, valid_pr-%.4f" \
                    % (epoch, epoch_loss, epoch_rmse, valid_rmse, valid_pr)
            logger.info(msg)
            scheduler.step(valid_rmse)
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
        valid_rmse, valid_pr = val(model, valid_loader, device)
        test2013_rmse, test2013_pr = val(model, test2013_loader, device)
        test2016_rmse, test2016_pr = val(model, test2016_loader, device)
        test2019_rmse, test2019_pr = val(model, test2019_loader, device)

        msg = "valid_rmse-%.4f, valid_pr-%.4f, test2013_rmse-%.4f, test2013_pr-%.4f, test2016_rmse-%.4f, test2016_pr-%.4f, test2019_rmse-%.4f, test2019_pr-%.4f," \
                    % (valid_rmse, valid_pr, test2013_rmse, test2013_pr, test2016_rmse, test2016_pr, test2019_rmse, test2019_pr)
        test2013_rmse, test2013_pr = val(model, test2013_loader, device)
        test2016_rmse, test2016_pr = val(model, test2016_loader, device)
        test2019_rmse, test2019_pr = val(model, test2019_loader, device)
        test2013_list.append(test2013_rmse)
        test2016_list.append(test2016_rmse)
        test2019_list.append(test2019_rmse)
        test2013_p.append(test2013_pr)
        test2016_p.append(test2016_pr)
        test2019_p.append(test2019_pr)
        print(test2013_list)
        print(test2016_list)
        print(test2019_list)
        print(test2013_p)
        print(test2016_p)
        print(test2019_p)
        logger.info(msg)
        logger.info("mean:" + str(np.mean(np.array(test2013_list))))
        logger.info("std:"+ str(np.std(np.array(test2013_list))))

# %%