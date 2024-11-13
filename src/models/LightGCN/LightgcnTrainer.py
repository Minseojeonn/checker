import torch
from tqdm import tqdm
from utils import log_param
from loguru import logger
from itertools import chain
from copy import deepcopy

class LightgcnTrainer:
    def __init__(
            self
        ):
        pass
        
    def train_with_hyper_param(self, num_nodes, train_data, val_data, hyper_param, evaluator, verbose=False, decoder_data=None):
        #hyper parameters
        self.epochs = hyper_param['epochs']
        self.learning_rate = hyper_param['learning_rate']
        self.in_dim = hyper_param['in_dim']
        self.out_dim = hyper_param['out_dim']
        self.layer_num = hyper_param['layer_num']

        #model init
        self.model = SDGNN(
            device=self.device,
            node_num=num_nodes, 
            edge_index_s=train_data.to(self.device),
            in_dim=hyper_param['in_dim'],
            out_dim=hyper_param['out_dim'],
            layer_num=hyper_param['layer_num'],
            init_emb=self.init_emb,
            init_emb_grad=self.init_emb_grad,
            lamb_d=self.lamb_d,
            lamb_t=self.lamb_t
        ).to(self.device)
        
        #score check
        best_score = 0
        
        #training phase
        pbar = tqdm(range(self.epochs), leave=False, colour='green', desc='epoch')
        for epoch in pbar:
            #train
            self.model.train()
            self.decoder.train()
            optimizer.zero_grad()
            loss = self.sign_prediction(train_data)
            loss.backward()
            optimizer.step()
            
            if verbose:
                pbar.write('Epoch {:02}: {:.4} training loss'.format(epoch, loss.item()))
            
            #eval
            with torch.no_grad():
                self.model.eval()
                self.decoder.eval()
                recall, hit = evaluator.evaluate(self.model.get_x(), val_data, train_data, K=10)
                #디버깅~
                score = recall
                print(recall, hit)
            
            if best_score < score:
                best_score = score
                torch.save(self.model.state_dict(), "./temp_best_model.pth")
                torch.save(self.decoder.state_dict(), "./temp_best_decoder.pth")
            
        pbar.close()
        #return best model.
        self.model.load_state_dict(torch.load("./temp_best_model.pth"))
        self.decoder.load_state_dict(torch.load("./temp_best_decoder.pth"))
        
        return self.model, self.decoder

    def sign_prediction(self, edges):
        embedding = self.model()
        logits = self.decoder(embedding, edges[:,:2])
        labels = (edges[:,2]==1).to(torch.float)
        loss = self.lossfn(logits.squeeze(), labels)
        return loss
    

    
def run_lightgcn(device, dataset, hyper_param, evaluator):
    #trainer
    trainer = LightgcnTrainer(device=device, init_emb=torch.randn((dataset.get_num_nodes(), hyper_param['in_dim']), device=device))
    model, decoder = trainer.train_with_hyper_param(
        num_nodes=dataset.get_num_nodes(),
        train_data=dataset.get_train(device),
        val_data=dataset.get_val_adj(),
        hyper_param=hyper_param,
        verbose=False,
        evaluator=evaluator
        )
    
    recall, hit = evaluator.evaluate(model.get_x(), dataset.get_test_adj(), dataset.get_train(device), K=10)
    
    return recall, hit

