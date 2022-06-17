import torch.nn as nn
import torch, tqdm, os

class FitterPipeline:
    def __init__(self, model, lossfunc, optimizer, device='cpu'):
        self.device = device
        self.model = model.to(self.device)
        self.lossfunc = lossfunc
        self.optimizer = optimizer
        
        self.model.apply(self.init_weights)
        
    
    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if torch.is_tensor(m.bias):
                m.bias.data.fill_(0.01)
    
    
    def save_model(self, dirname='./model_params', 
                   filename='tumor_segmentation_params.pth.tar'):
        if not os.path.isdir(dirname):os.mkdir(dirname)
        state_dicts = {
            'segmentation_net_params':self.model.state_dict(),
            'optimizer_params':self.optimizer.state_dict(),
        }
        return torch.save(state_dicts, os.path.join(dirname, filename))
        

    def train(self, dataloader, epochs=1, verbose=False):
        self.model.train()
        
        losses, hds, dice_coeffs = [], [], []
        for epoch in range(epochs):
            loss, hd, dice_coeff = 0, 0, 0
            for idx, (image, gt_mask) in tqdm.tqdm(enumerate(dataloader)):
                
                self.model.zero_grad()
                image, gt_mask = image.to(self.device), gt_mask.to(self.device)
                
                pred_mask = self.model(image)
                
                #batch loss
                batch_loss = self.lossfunc(pred_mask, gt_mask)

                #backward propagation
                batch_loss.backward()
                self.optimizer.step()
                
                #metric values
                batch_hd, batch_dice_coeff, _ = self.lossfunc.metric_values(pred_mask, gt_mask)

                loss += batch_loss.item()
                dice_coeff += batch_dice_coeff
                hd += batch_hd
            
            loss = loss / (idx + 1)
            hd = hd / (idx + 1)
            dice_coeff = dice_coeff / (idx + 1)
            
            if verbose:
                print(f'epoch: {epoch} \nTraining Loss: {loss} \nTrain Dice Coeff: {dice_coeff} \nTrain Hausdorff Distance: {hd}')
            losses.append(loss)
            hds.append(hd)
            dice_coeffs.append(dice_coeff)
        return losses, hds, dice_coeffs
    
    
    def test(self, dataloader, verbose=False):
        self.model.eval()
        
        loss, hd, dice_coeff = 0, 0, 0
        with torch.no_grad():
            for idx, (image, gt_mask) in tqdm.tqdm(enumerate(dataloader)):

                image, gt_mask = image.to(self.device), gt_mask.to(self.device)

                pred_mask = self.model(image)

                #batch loss
                batch_loss = self.lossfunc(pred_mask, gt_mask)
                                
                #metric values
                batch_hd, batch_dice_coeff, _ = self.lossfunc.metric_values(pred_mask, gt_mask)

                loss += batch_loss.item()
                dice_coeff += batch_dice_coeff
                hd += batch_hd
            
        loss = loss / (idx + 1)
        hd = hd / (idx + 1)
        dice_coeff = dice_coeff / (idx + 1)

        if verbose:
            print(f'\n Testing Loss: {loss}  \nTest Dice Coeff: {dice_coeff} \nTest Hausdorff Distance: {hd}')
        return loss, hd, dice_coeff