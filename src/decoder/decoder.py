import torch
from loguru import logger
class SignDecoder(torch.nn.Module):
    def __init__(self, emb_dim, decoder_dim, device):
        """
        Parameters
        ----------
        emb_dim
            dimension size of the model's output embedding feature
        decoder_dim
            hidden classifier dimension
        device
            device name
        """
        super().__init__()

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_dim, decoder_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(decoder_dim, 1)
        ).to(device)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, embedding, indices):
        """
        Parameters
        ----------
        embedding
            embedding feature
        indices
            list of the edge which is index of the node

        Returns
        -------
        Logits respect to given embedding and indices
        """
        fr = embedding[indices[:,0]]
        to = embedding[indices[:,1]]
        logits = self.mlp(torch.concat((fr,to),dim=1))
        
        return self.sigmoid(logits)
    
class DotProduct(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embedding, indices):
        """
        Parameters
        ----------
        embedding
            embedding feature
        indices
            list of the edge which is index of the node

        Returns
        -------
        Logits respect to given embedding and indices
        """
        fr = embedding[indices[:,0]]
        to = embedding[indices[:,1]]
        return torch.sum(fr * to, dim=1)