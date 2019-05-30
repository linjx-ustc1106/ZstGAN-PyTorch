from model import *
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import itertools



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if len(output[0]) < topk[1]:
        topk = (1, len(output[0]))
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class Solver(object):
    """Solver for training and testing zstgan."""

    def __init__(self, data_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.data_loader = data_loader

        # Model configurations.
        self.ft_num = config.ft_num
        self.nz_num = config.nz_num
        self.c_dim = config.c_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.n_blocks = config.n_blocks
        self.lambda_mut = config.lambda_mut     
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.att_num = config.att_num

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.ev_ea_c_iters = config.ev_ea_c_iters
        self.c_pre_iters = config.c_pre_iters

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model 
        self.build_model()

    def build_model(self):
        """Create networks."""

        self.encoder = AdaINEnc(input_dim = 3, ft_num = self.ft_num)
        self.decoder = AdaINDec(input_dim = 3, ft_num = self.ft_num)
        self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num, ft_num = self.ft_num) 
        self.encoder_v = Resnet_Feature()
        self.encoder_a = MLP_Encoder(in_dim = self.att_num, nz_num = self.nz_num, out_dim= self.ft_num)
        self.D_s = Eb_Discriminator(ft_num = self.ft_num, att_num = self.att_num)
        self.C = Linear_Classifier(in_dim= self.ft_num, c_dim = self.c_dim)
        
        self.g_optimizer = torch.optim.Adam(itertools.chain(self.encoder.parameters(), self.decoder.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.ev_optimizer = torch.optim.Adam(itertools.chain(self.encoder_v.parameters(), self.C.parameters()), self.d_lr, [self.beta1, self.beta2]) # use the same optimizer to update encoder_v and C
        self.ea_optimizer = torch.optim.Adam(self.encoder_a.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.ds_optimizer = torch.optim.Adam(self.D_s.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.c_optimizer = torch.optim.Adam(self.C.parameters(), self.d_lr, [self.beta1, self.beta2])
        
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.D.to(self.device)
        self.encoder_v.to(self.device)
        self.encoder_a.to(self.device)
        self.D_s.to(self.device)
        self.C.to(self.device)
        
        
        

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained networks."""
        
        print('Loading the trained models from step {}...'.format(resume_iters))
        encoder_path = os.path.join(self.model_save_dir, '{}-encoder.ckpt'.format(resume_iters))
        decoder_path = os.path.join(self.model_save_dir, '{}-decoder.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()


    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out


    def classification_loss(self, logit, target):
        """Compute softmax cross entropy loss."""
        return F.cross_entropy(logit, target)

       
    def train_ev_ea(self):
        """Train encoder_a and encoder_v with C and D_s."""
        # Set data loader.
        data_loader = self.data_loader
        
        noise = torch.FloatTensor(self.batch_size, self.nz_num)
        noise = noise.to(self.device) # noise vector z
       
        start_iters = 0

        # Start training.
        print('Start encoder_a and encoder_v training...')
        start_time = time.time()
        
        ev_ea_c_iters = self.ev_ea_c_iters
        c_pre_iters = self.c_pre_iters
        
        C_path = os.path.join(self.model_save_dir, '{}-C.ckpt'.format(ev_ea_c_iters))
        
        encoder_a_path = os.path.join(self.model_save_dir, '{}-encoder_a.ckpt'.format(ev_ea_c_iters))
        
        encoder_v_path = os.path.join(self.model_save_dir, '{}-encoder_v.ckpt'.format(ev_ea_c_iters))
                
        
        if os.path.exists(C_path):
            self.C.load_state_dict(torch.load(C_path, map_location=lambda storage, loc: storage))
            print('Load model checkpoints from {}'.format(C_path))
            
            self.encoder_a.load_state_dict(torch.load(encoder_a_path, map_location=lambda storage, loc: storage))
            print('Load model checkpoints from {}'.format(encoder_a_path))
            
            self.encoder_v.load_state_dict(torch.load(encoder_v_path, map_location=lambda storage, loc: storage))
            print('Load model checkpoints from {}'.format(encoder_v_path))
        else:
            C_pre_path = os.path.join(self.model_save_dir, '{}-C.ckpt'.format(c_pre_iters))
            if os.path.exists(C_pre_path):
                self.C.load_state_dict(torch.load(C_pre_path, map_location=lambda storage, loc: storage))
                print('Load model pretrained checkpoints from {}'.format(C_pre_path))
            else:
                for i in range(0, c_pre_iters):
                    # Fetch real images, attributes and labels.
                    x_real, wrong_images, attributes, _, label_org = data_loader.train.next_batch(self.batch_size,10)


                    x_real = x_real.to(self.device)           # Input images.
                    attributes = attributes.to(self.device)   # Input attributes
                    label_org = label_org.to(self.device)     # Labels for computing classification loss.
                    
                    ev_x = self.encoder_v(x_real)
                    cls_x = self.C(ev_x.detach())
                    # Classification loss from only images for C training
                    c_loss_cls = self.classification_loss(cls_x, label_org) 
                    # Backward and optimize.
                    self.c_optimizer.zero_grad()
                    c_loss_cls.backward()
                    self.c_optimizer.step()
                    
                    if (i+1) % self.log_step == 0:
                        loss = {}
                        loss['c_loss_cls'] = c_loss_cls.item()
                        prec1, prec5 = accuracy(cls_x.data, label_org.data, topk=(1, 5))
                        loss['prec1'] = prec1
                        loss['prec5'] = prec5
                        log = "C pretraining iteration [{}/{}]".format(i+1, c_pre_iters)
                        for tag, value in loss.items():
                            log += ", {}: {:.4f}".format(tag, value)
                        print(log)
                torch.save(self.C.state_dict(), C_pre_path)
                print('Saved model pretrained checkpoints into {}...'.format(C_pre_path))
           
            for i in range(c_pre_iters, ev_ea_c_iters):
                # Fetch real images, attributes and labels.
                x_real, wrong_images, attributes, _, label_org = data_loader.train.next_batch(self.batch_size,10)


                x_real = x_real.to(self.device)           # Input images.
                attributes = attributes.to(self.device)   # Input attributes
                label_org = label_org.to(self.device)     # Labels for computing classification loss.
                   

                # =================================================================================== #
                #       Train the domain-specific features discriminator                              
                # =================================================================================== #
                
                noise.normal_(0, 1)
                # Compute embedding of both images and attributes
                ea_a = self.encoder_a(attributes, noise)
                ev_x = self.encoder_v(x_real)
                
                
                ev_x_real = self.D_s(ev_x, attributes)
                ds_loss_real = -torch.mean(ev_x_real)
                
                
                ea_a_fake = self.D_s(ea_a, attributes)
                ds_loss_fake = torch.mean(ea_a_fake)
                
                # Compute loss for gradient penalty.
                alpha = torch.rand(ev_x.size(0), 1).to(self.device)
                ebd_hat = (alpha * ev_x.data + (1 - alpha) * ea_a.data).requires_grad_(True)
                
                ebd_inter = self.D_s(ebd_hat, attributes)
                ds_loss_gp = self.gradient_penalty(ebd_inter, ebd_hat)
                
                ds_loss = ds_loss_real + ds_loss_fake + self.lambda_gp * ds_loss_gp #+ ds_loss_realw
                #self.reset_grad_eb()
                self.ea_optimizer.zero_grad()
                self.ds_optimizer.zero_grad()
                self.ev_optimizer.zero_grad()

                ds_loss.backward()
                self.ds_optimizer.step()
                if (i+1) % self.n_critic == 0:
                    # =================================================================================== #
                    #                              Train the encoder_a and C                              
                    # =================================================================================== #
                    ev_x = self.encoder_v(x_real)
                    ev_x_real = self.D_s(ev_x, attributes)
                    ev_loss_real = torch.mean(ev_x_real)
                    
                    cls_x = self.C(ev_x)
                    c_loss_cls = self.classification_loss(cls_x, label_org)

                    # Backward and optimize.
                    ev_c_loss = ev_loss_real + c_loss_cls
                    self.ea_optimizer.zero_grad()
                    self.ds_optimizer.zero_grad()
                    self.ev_optimizer.zero_grad()
                    ev_c_loss.backward()
                    self.ev_optimizer.step()
                    
                    # =================================================================================== #
                    #                              Train the encoder_v                              #
                    # =================================================================================== #
                    noise.normal_(0, 1)
                    ea_a = self.encoder_a(attributes,noise)
                    ea_a_fake = self.D_s(ea_a, attributes)
                    ea_loss_fake = -torch.mean(ea_a_fake)
                    
                    cls_a = self.C(ea_a)
                    ebn_loss_cls = self.classification_loss(cls_a, label_org)
                    

                    # Backward and optimize.
                    ea_loss = ea_loss_fake + ebn_loss_cls
                    self.ea_optimizer.zero_grad()
                    self.ds_optimizer.zero_grad()
                    self.ev_optimizer.zero_grad()
                    ea_loss.backward()
                    self.ea_optimizer.step()
                   
                    # Logging.
                    loss = {}
                     
                    loss['ds/ds_loss_real'] = ds_loss_real.item()
                    loss['ds/ds_loss_fake'] = ds_loss_fake.item()
                    loss['ds/ds_loss_gp'] = ds_loss_gp.item()
               
                # Print out training information.
                if (i+1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    prec1, prec5 = accuracy(cls_x.data, label_org.data, topk=(1, 5))
                    loss['prec1'] = prec1
                    loss['prec5'] = prec5
                    prec1e, prec5e = accuracy(cls_a.data, label_org.data, topk=(1, 5))
                    loss['prec1e'] = prec1e
                    loss['prec5e'] = prec5e
                    log = "Encoder_a and Encoder_v Training Elapsed [{}], Iteration [{}/{}]".format(et, i+1, ev_ea_c_iters)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

              
                # Save model checkpoints.
                if (i+1) % self.model_save_step == 0:
                    C_path = os.path.join(self.model_save_dir, '{}-C.ckpt'.format(i+1))
                    torch.save(self.C.state_dict(), C_path)
                    print('Saved model checkpoints into {}...'.format(C_path))
                    
                    encoder_a_path = os.path.join(self.model_save_dir, '{}-encoder_a.ckpt'.format(i+1))
                    torch.save(self.encoder_a.state_dict(), encoder_a_path)
                    print('Saved model checkpoints into {}...'.format(encoder_a_path))
                    
                    encoder_v_path = os.path.join(self.model_save_dir, '{}-encoder_v.ckpt'.format(i+1))
                    torch.save(self.encoder_v.state_dict(), encoder_v_path)
                    print('Saved model checkpoints into {}...'.format(encoder_v_path))
                
    def train(self):
        """Train zstgan"""
        # train encoder_a and encoder_v first
        self.train_ev_ea()
        self.encoder_v.eval()
        
        # Set data loader.
        data_loader = self.data_loader
       
        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr
        
        # noise vector z
        noise = torch.FloatTensor(self.batch_size, self.nz_num)
        noise = noise.to(self.device)

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)
       
        # Start training.
        print('Start training...')
        start_time = time.time()
        empty = torch.FloatTensor(1, 3,self.image_size,self.image_size).to(self.device) 
        empty.fill_(1)
        for i in range(start_iters, self.num_iters):
            # Fetch real images and labels.
            x_real, wrong_images, attributes, _, label_org = data_loader.train.next_batch(self.batch_size,10)
            label_org = label_org.to(self.device)  
            attributes = attributes.to(self.device) 
            x_real = x_real.to(self.device)         
            # Generate target domains
            ev_x = self.encoder_v(x_real) 
            
            rand_idx = torch.randperm(label_org.size(0))
            
            trg_ev_x_1 = ev_x[rand_idx]
            trg_ev_x = trg_ev_x_1.clone()
            label_trg_1 = label_org[rand_idx]
            label_trg = label_trg_1.clone()

            # =================================================================================== #
            #                             Train the discriminator                              
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_mut = torch.mean(torch.abs(ev_x.detach() - out_cls))

            # Compute loss with fake images.
            x_fake = self.decoder(self.encoder(x_real), trg_ev_x)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_mut * d_loss_mut + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_mut'] = d_loss_mut.item()
            loss['D/loss_gp'] = d_loss_gp.item()
           
            # =================================================================================== #
            #                               Train the encoder and decoder                                
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                x_di = self.encoder(x_real)
                
                x_fake = self.decoder(x_di, trg_ev_x)
                x_reconst1 = self.decoder(x_di, ev_x)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_mut = torch.mean(torch.abs(trg_ev_x.detach() - out_cls)) 

                # Target-to-original domain.
                x_fake_di = self.encoder(x_fake)
                
                x_reconst2 = self.decoder(x_fake_di, ev_x)
                
                g_loss_rec1 = torch.mean(torch.abs(x_real - x_reconst1))
                
                g_loss_rec12 = torch.mean(torch.abs(x_real - x_reconst2))

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * (g_loss_rec1 + g_loss_rec12) + self.lambda_mut * g_loss_mut
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec1'] = g_loss_rec1.item()
                loss['G/loss_rec2'] = g_loss_rec12.item()
                loss['G/loss_mut'] = g_loss_mut.item()

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    out_A2B_results = [empty]

                    for idx1 in range(label_org.size(0)):
                        out_A2B_results.append(x_real[idx1:idx1+1])

                    for idx2 in range(label_org.size(0)):
                        out_A2B_results.append(x_real[idx2:idx2+1])
                        
                        for idx1 in range(label_org.size(0)):
                            x_fake = self.decoder(self.encoder(x_real[idx2:idx2+1]), ev_x[idx1:idx1+1])
                            out_A2B_results.append(x_fake)
                    results_concat = torch.cat(out_A2B_results)
                    x_AB_results_path = os.path.join(self.sample_dir, '{}_x_AB_results.jpg'.format(i+1)) 
                    save_image(self.denorm(results_concat.data.cpu()), x_AB_results_path, nrow=label_org.size(0)+1,padding=0)
                    print('Saved real and fake images into {}...'.format(x_AB_results_path))
                    # save vision-driven and attribute-driven results on unseen domains
                    x_real, wrong_images, attributes, _, label_org = data_loader.test.next_batch(self.batch_size,10)
                    label_org = label_org.to(self.device)  
                    x_real = x_real.to(self.device) 
                    attributes = attributes.to(self.device) 
                    ev_x = self.encoder_v(x_real)
                    noise.normal_(0, 1)
                    ea_a = self.encoder_a(attributes, noise)
                    
                    out_A2B_results = [empty]
                    out_A2B_results_a = [empty]

                    for idx1 in range(label_org.size(0)):
                        out_A2B_results.append(x_real[idx1:idx1+1])
                        out_A2B_results_a.append(x_real[idx1:idx1+1])

                    for idx2 in range(label_org.size(0)):
                        out_A2B_results.append(x_real[idx2:idx2+1])
                        out_A2B_results_a.append(x_real[idx2:idx2+1])
                        
                        for idx1 in range(label_org.size(0)):
                            x_fake = self.decoder(self.encoder(x_real[idx2:idx2+1]), ev_x[idx1:idx1+1])
                            out_A2B_results.append(x_fake)
                            
                            x_fake_a = self.decoder(self.encoder(x_real[idx2:idx2+1]), ea_a[idx1:idx1+1])
                            out_A2B_results_a.append(x_fake_a)
                    results_concat = torch.cat(out_A2B_results)
                    x_AB_results_path = os.path.join(self.sample_dir, '{}_x_AB_results_test_v.jpg'.format(i+1)) 
                    save_image(self.denorm(results_concat.data.cpu()), x_AB_results_path, nrow=label_org.size(0)+1,padding=0)
                    print('Saved real and fake images into {}...'.format(x_AB_results_path))
                    
                    results_concat = torch.cat(out_A2B_results_a)
                    x_AB_results_path = os.path.join(self.sample_dir, '{}_x_AB_results_test_a.jpg'.format(i+1)) 
                    save_image(self.denorm(results_concat.data.cpu()), x_AB_results_path, nrow=label_org.size(0)+1,padding=0)
                    print('Saved real and fake images into {}...'.format(x_AB_results_path))
                



            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                encoder_path = os.path.join(self.model_save_dir, '{}-encoder.ckpt'.format(i+1))
                decoder_path = os.path.join(self.model_save_dir, '{}-decoder.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.encoder.state_dict(), encoder_path)
                torch.save(self.decoder.state_dict(), decoder_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))
                
                

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))


    
    def test(self):
        """Translate images using zstgan on unseen test set."""
        # Load the trained models.
        self.train_ev_ea()
        self.restore_model(self.test_iters)
        self.encoder_v.eval()
        # Set data loader.
        data_loader = self.data_loader
        empty = torch.FloatTensor(1, 3,self.image_size,self.image_size).to(self.device) 
        empty.fill_(1)
        noise = torch.FloatTensor(self.batch_size, self.nz_num)
        noise = noise.to(self.device)
        step = 0
        data_loader.test.reinitialize_index()
        with torch.no_grad():
            while True:
                try:
                    x_real, wrong_images, attributes, _, label_org = data_loader.test.next_batch_test(self.batch_size,10)
                except:
                    break
                x_real = x_real.to(self.device)         
                label_org = label_org.to(self.device)
                attributes = attributes.to(self.device)
                
                
                ev_x = self.encoder_v(x_real)
                noise.normal_(0, 1)
                ea_a = self.encoder_a(attributes, noise)
                
                out_A2B_results = [empty]
                out_A2B_results_a = [empty]

                for idx1 in range(label_org.size(0)):
                    out_A2B_results.append(x_real[idx1:idx1+1])
                    out_A2B_results_a.append(x_real[idx1:idx1+1])

                for idx2 in range(label_org.size(0)):
                    out_A2B_results.append(x_real[idx2:idx2+1])
                    out_A2B_results_a.append(x_real[idx2:idx2+1])
                    
                    for idx1 in range(label_org.size(0)):
                        x_fake = self.decoder(self.encoder(x_real[idx2:idx2+1]), ev_x[idx1:idx1+1])
                        out_A2B_results.append(x_fake)
                        
                        x_fake_a = self.decoder(self.encoder(x_real[idx2:idx2+1]), ea_a[idx1:idx1+1])
                        out_A2B_results_a.append(x_fake_a)
                results_concat = torch.cat(out_A2B_results)
                x_AB_results_path = os.path.join(self.result_dir, '{}_x_AB_results_test_v.jpg'.format(step+1)) 
                save_image(self.denorm(results_concat.data.cpu()), x_AB_results_path, nrow=label_org.size(0)+1,padding=0)
                print('Saved real and fake images into {}...'.format(x_AB_results_path))
                
                results_concat = torch.cat(out_A2B_results_a)
                x_AB_results_path = os.path.join(self.result_dir, '{}_x_AB_results_test_a.jpg'.format(step+1)) 
                save_image(self.denorm(results_concat.data.cpu()), x_AB_results_path, nrow=label_org.size(0)+1,padding=0)
                print('Saved real and fake images into {}...'.format(x_AB_results_path))
                
                step += 1
                
                
               
                
                
   