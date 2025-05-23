import os
import pickle
import torch

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def create_dir(dir_list):
    assert  isinstance(dir_list, list) == True
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)

def save_model_dict(model, model_dir, msg):
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(model.state_dict(), model_path)
    print("model has been saved to %s." % (model_path))

def load_model_dict(model, ckpt):
    model.load_state_dict(torch.load(ckpt))

def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path,i)  
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)

def write_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

class BestMeter(object):
    """Computes and stores the best value"""

    def __init__(self, best_type):
        self.best_type = best_type  
        self.count = 0      
        self.reset()

    def reset(self):
        if self.best_type == 'min':
            self.best = float('inf')
        else:
            self.best = -float('inf')

    def update(self, best):
        self.best = best
        self.count = 0

    def get_best(self):
        return self.best

    def counter(self):
        self.count += 1
        return self.count

import matplotlib.pyplot as plt


def plot_training_curve(line_1_x, line_1_y, line_2_x, line_2_y, save_path, x_label='Epoch', y_label='Loss'):
	plt.plot(line_1_x, line_1_y, 'o-g', label='Train Dataset', markersize=5)
	plt.plot(line_2_x, line_2_y, 'o-r', label='Valid Dataset', markersize=5)
	plt.xlabel(x_label, size=14)
	plt.ylabel(y_label, size=14)
	plt.title('Training Curve', size=16)
	plt.xticks(size=10)
	plt.yticks(size=10)
	if(line_1_x[0]>3):
	    plt.ylim(0,x[0]+1)
	    
	else:
	    plt.ylim(0, 3)
	plt.legend(loc='lower left', fontsize=15)
	plt.savefig(save_path, bbox_inches="tight")
	plt.clf()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return self.avg
