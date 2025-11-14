import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
import pickle

class ProteinDataset(Dataset):
    def __init__(self, coords_path, reps_path, layer=-1, device='cuda'):
        self.device = device
        self.layer = layer

        with open(reps_path, 'rb') as f:
            self.representation_list = pickle.load(f)
        with open(coords_path, 'rb') as f:
            self.coords_list = pickle.load(f)

    def __len__(self):
        return len(self.representation_list)

    def __getitem__(self, idx):
        rep = self.representation_list[self.layer][idx]
        coords = self.coords_list[idx]

        rep = torch.tensor(rep, dtype=torch.float32, device=self.device)
        coords = torch.tensor(coords, dtype=torch.float32, device=self.device)
        return coords, rep


class FoldingHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim,hidden_dim//2),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim//2,3))
        
    def forward(self,x):
        return self.net(x)

#%% training loop for ESM2 150M
train_losses = [[] for i in range(30)]
test_losses = [[] for i in range(30)]
epochs = 500
for layer in range(30):
    dataset = ProteinDataset('../data/reps/coords_space.pickle', 
                             '../data/reps/coords_esm_space_esm2_t30_150M_UR50D_.pickle',
                             layer=layer)
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    head = FoldingHead(input_dim = dataset.representation_list[layer][0].shape[-1]).cuda()
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in tqdm(range(epochs)):
        train_loss = 0
        for coords, rep in train_loader:
            rep = rep.squeeze(0).to('cuda')
            coords = coords.squeeze(0).to('cuda')

            coords_pred = head(rep)
            loss = loss_fn(coords_pred, coords)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch==epochs-1:
                train_loss += loss.item()
                
        if epoch%100==0:
            print(loss.item())
            
    train_losses[layer] = train_loss/len(train_loader)
    head.eval()
    test_loss = 0
    with torch.no_grad():
        for coords, rep in test_loader:
            rep = rep.squeeze(0).to('cuda')
            coords = coords.squeeze(0).to('cuda')
    
            coords_pred = head(rep)
            loss = loss_fn(coords_pred, coords)
            test_loss += loss.item()
            
    test_losses[layer] = test_loss/len(test_loader) 
    print(f'Done with layer {layer}')
    
with open('../data/layerwise_train_losses.pickle', 'wb') as f:
            pickle.dump(train_losses, f)
with open('../data/layerwise_test_losses.pickle', 'wb') as f:
            pickle.dump(test_losses, f)



