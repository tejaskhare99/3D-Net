import argparse
import torch 
from torch.utils.data import DataLoader
from training import train
from dataloader import Dataset2d_3d
from network import CNN_Autoencoder

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str,  default='train')
parser.add_argument('--epochs',type=int,default=50)

def train_model(args,device):
    dataset = Dataset2d_3d()
    train_loader = DataLoader(dataset,pin_memory=True,shuffle=True,batch_size=64)
    model = CNN_Autoencoder()
    print(model)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    train(model, train_loader, optimizer , criterion , device, args.epochs)



def run_inference():
    pass

def evaluate_model(path):
    pass

if __name__ == '__main__':
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    if(args.mode=='train'):
        train_model(args,device)








