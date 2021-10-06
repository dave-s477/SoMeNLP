import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import Dataset
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter

class DisambiguationModel(nn.Module):
    def __init__(self, dim, layer_sizes, drop_outs):
        super(DisambiguationModel, self).__init__()
        self.lin_layers = nn.ModuleList([nn.Linear(dim, layer_sizes[0])])
        for i in range(len(layer_sizes)-1):
            self.lin_layers.append(nn.ReLU())
            self.lin_layers.append(nn.Dropout(drop_outs[i]))
            self.lin_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

    def forward(self, x):
        for l in self.lin_layers:
            x = l(x)
        return x

class ModelWrapper():
    def __init__(self, config, input_dim, save_path=None, device='cpu'):
        self.config = config
        self.device = device
        self.writer = SummaryWriter('{}/log'.format(save_path))
        self.model = DisambiguationModel(input_dim, config['layer_sizes'], config['drop_outs']).to(device)
        self.loss_fn = nn.BCEWithLogitsLoss().to(device)
        self.optim = optim.Adam(self.model.parameters())
        self.epoch = 0
        self.save_path = save_path

    def train(self, train_set):
        self.model.train()
        cum_loss = 0
        for idx, sample in enumerate(train_set):
            self.optim.zero_grad()
            pred = self.model(sample[0].to(self.device))
            loss = self.loss_fn(torch.squeeze(pred), sample[1].float().to(self.device))
            loss.backward()
            self.optim.step()
            cum_loss += loss.detach()
            if idx != 0 and idx % 4000 == 0:
                print("Batch {} Avg. Loss {}".format(idx, cum_loss / 100)) 
                cum_loss = 0

    def test(self, test_set):
        self.model.eval()
        predictions = []
        true = []
        for idx, sample in enumerate(test_set):
            with torch.no_grad():
                pred = self.model(sample[0].to(self.device))
            pred_class = torch.squeeze(torch.sigmoid(pred))
            predictions.extend(pred_class.cpu().numpy())
            true.extend(sample[1])
            if idx != 0 and idx % 4000 == 0:
                print("At batch {}".format(idx))
        return true, predictions

    def predict(self, test_set='train', test_set_ext='train'):
        self.model.eval()
        predictions = []
        for idx, sample in enumerate(self.inputs[test_set][test_set_ext]['loader']):
            with torch.no_grad():
                pred = self.model(sample[0])
            pred_class = torch.squeeze(torch.sigmoid(pred))
            predictions.extend(pred_class.cpu().numpy())
        return predictions

    def eval(self, true, predictions, threshold=.5, epoch=0, write=True):
        predictions = [0 if pred <= threshold else 1 for pred in predictions]
        print(classification_report(true, predictions))
        classification_dict = classification_report(true, predictions, output_dict=True)
        if write and 'True' in classification_dict:
            for k,v in classification_dict['True'].items():
                self.writer.add_scalar(k, v, epoch)

    def save(self):
        print("Saving model")
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
        }, '{}/model.pth'.format(self.save_path))

    def load(self):
        if 'checkpoint' in self.config and self.config['checkpoint']:
            print("Loading model from checkpoint")
            checkpoint_data = torch.load(self.config['checkpoint'], map_location=self.device)
            self.model.load_state_dict(checkpoint_data['model_state_dict'])
            self.optim.load_state_dict(checkpoint_data['optimizer_state_dict'])
            self.epoch = checkpoint_data['epoch'] 

