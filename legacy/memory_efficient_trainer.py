"""
Gradient accumulation trainer for memory efficiency
This allows using larger effective batch sizes with limited GPU memory
"""

import torch
from trainers.contrastive_pix2pix_trainer import ContrastivePix2PixTrainer

class MemoryEfficientContrastivePix2PixTrainer(ContrastivePix2PixTrainer):
    def __init__(self, opt):
        super().__init__(opt)
        self.accumulation_steps = getattr(opt, 'accumulation_steps', 4)
        self.accumulation_counter = 0
        
    def run_generator_one_step(self, data, iter=0):
        """Modified generator step with gradient accumulation"""
        self.netG.zero_grad()
        
        # Normalize loss by accumulation steps
        loss_G, loss_dict = self.compute_generator_loss(data)
        loss_G = loss_G / self.accumulation_steps
        loss_G.backward()
        
        self.accumulation_counter += 1
        
        # Only update weights after accumulating gradients
        if self.accumulation_counter % self.accumulation_steps == 0:
            self.optimizer_G.step()
            self.optimizer_G.zero_grad()
            
        # Store losses (multiply back by accumulation steps for logging)
        self.loss_dict = {k: v * self.accumulation_steps for k, v in loss_dict.items()}
        
    def run_discriminator_one_step(self, data):
        """Modified discriminator step with gradient accumulation"""
        self.netD.zero_grad()
        
        # Normalize loss by accumulation steps
        loss_D, loss_dict = self.compute_discriminator_loss(data)
        loss_D = loss_D / self.accumulation_steps
        loss_D.backward()
        
        # Only update weights after accumulating gradients
        if self.accumulation_counter % self.accumulation_steps == 0:
            self.optimizer_D.step()
            self.optimizer_D.zero_grad()
            
        # Update loss dict
        for k, v in loss_dict.items():
            if k in self.loss_dict:
                self.loss_dict[k] += v * self.accumulation_steps
            else:
                self.loss_dict[k] = v * self.accumulation_steps