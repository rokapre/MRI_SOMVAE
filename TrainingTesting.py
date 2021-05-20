import numpy as np 
import torch
import torch.nn as nn
import time, datetime


"""
Trains a VAE model with a given optimizer, loss function, training data

Taken from https://github.com/RdoubleA/DWI-inpainting

Arguments
    - model: torch.nn.module
    - optimizer: one of the pytorch optimizers
    - loss_function: pointer to a function to compute model loss, should take in arguments 
      (original image, reconstructed image, latent mean, latent logvariance)
    - training_generator: pytorch DataLoader object that is a generator to yield batches of training data
    - epoch: integer specifying which epoch we're on, zero indexed. sole purpose is for printing progress
    - log_every_num_batches: integere specify after how many batches do we print progress
"""
def train(model, optimizer, loss_function, training_generator, epoch, log_every_num_batches = 40):
    print('====> Begin epoch {}'.format(epoch+1))
    print()
    # Setup GPU if available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    if use_cuda:
        if torch.cuda.device_count() > 1:
          print("Let's use", torch.cuda.device_count(), "GPUs!")
          # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
          model = nn.DataParallel(model)
    model.to(device)

    t0 = time.time()
    model.train()
    train_loss = 0
    batch_id = 1
    batch_size = training_generator.batch_size
    for batch_in, batch_out in training_generator:
        batch_run_time = time.time()
        # Transfer to GPU
        batch_in, batch_out = batch_in.to(device), batch_out.to(device)
        
        # Clear optimizer gradients
        optimizer.zero_grad()
        # Forward pass through the model
        outputs = model(batch_in)
        
        # Calculate loss
        recon_loss = loss_function(outputs['x_out'], batch_out)
        if 'vq_loss' in outputs:
            loss = recon_loss + outputs['vq_loss']
        else:
            loss = recon_loss

        # Back propagate
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        batch_run_time = time.time() - batch_run_time
        et = batch_run_time * (len(training_generator) - batch_id) * 3
        # Print progress
        if batch_id % log_every_num_batches == 0 or batch_id == 1:
            print('Train Epoch: {:d} [{:d}/{:d} ({:.0f}%)]\tLoss: {:.6f}\tET - {:s}'.format(
                    epoch+1, batch_id, len(training_generator),
                    100. * batch_id / len(training_generator),
                    loss.item() / len(batch_in), str(datetime.timedelta(seconds=int(et)))))
        batch_id += 1
        
    t_epoch = time.time() - t0
    train_loss /= len(training_generator) * batch_size
    print()
    print('====> Epoch: {} Average loss: {:.4f}\tTime elapsed: {:s}'.format(
          epoch+1, train_loss, str(datetime.timedelta(seconds=int(t_epoch)))))
    return train_loss

"""
Evaluates a VAE model with a given validation/test set

From https://github.com/RdoubleA/DWI-inpainting

Arguments
    - model: torch.nn.module
    - loss_function: pointer to a function to compute model loss, should take in arguments 
      (original image, reconstructed image, latent mean, latent logvariance)
    - validation_generator: pytorch DataLoader object that is a generator to yield batches of test data
"""
def test(model, loss_function, validation_generator):
    # Setup GPU if available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    if use_cuda:
        if torch.cuda.device_count() > 1:
          print("Let's use", torch.cuda.device_count(), "GPUs!")
          # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
          model = nn.DataParallel(model)
    model.to(device)

    model.eval()
    test_loss = 0
    t0 = time.time()
    batch_size = validation_generator.batch_size
    with torch.no_grad():
        for batch_in, batch_out in validation_generator:
            # Transfer to GPU
            batch_in, batch_out = batch_in.to(device), batch_out.to(device)
            # Forward pass through the model
            outputs = model(batch_in)
            
            # Calculate loss
            recon_loss = loss_function(outputs['x_out'], batch_out)
            if 'vq_loss' in outputs:
                loss = recon_loss + outputs['vq_loss']
            else:
                loss = recon_loss
            test_loss += loss.item()
    
    t_epoch = time.time() - t0
    test_loss /= len(validation_generator) * batch_size
    print('====> Test set loss: {:.4f}\tTime elapsed: {:s}'.format(
        test_loss, str(datetime.timedelta(seconds=int(t_epoch)))))
    print()
    return test_loss


"""
Trains a NewVQVAE model with a given optimizer, loss function, training data

This is a modified form of the train() function above which is meant to be used for models with
both reconstructions from the encoder and the quantized layer.

Arguments
    - model: torch.nn.module
    - optimizer: one of the pytorch optimizers
    - loss_function: pointer to a function to compute model loss, should take in arguments 
      (original image, reconstructed image, latent mean, latent logvariance)
    - training_generator: pytorch DataLoader object that is a generator to yield batches of training data
    - epoch: integer specifying which epoch we're on, zero indexed. sole purpose is for printing progress
    - log_every_num_batches: integere specify after how many batches do we print progress
    - lam_ze: 0 <= lam_ze <= 1 Weighting for reconstruction resulting from ze 
    - lam_zq: 0 <= lam_zq <= 1 Weighting for reconstruction resulting from zq
"""
def train_NewVQVAE(model, optimizer, loss_function, training_generator, epoch, log_every_num_batches = 40,lam_ze = 0.5, lam_zq = 0.5):
    assert lam_ze >= 0 
    assert lam_zq >= 0 
    lam_sum = lam_ze + lam_zq
    assert lam_sum > 0 
    lam_ze /= lam_sum
    lam_zq /= lam_sum
    print('====> Begin epoch {}'.format(epoch+1))
    print()
    # Setup GPU if available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    if use_cuda:
        if torch.cuda.device_count() > 1:
          print("Let's use", torch.cuda.device_count(), "GPUs!")
          # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
          model = nn.DataParallel(model)
    model.to(device)

    t0 = time.time()
    model.train()
    train_loss = 0
    batch_id = 1
    batch_size = training_generator.batch_size
    for batch_in, batch_out in training_generator:
        batch_run_time = time.time()
        # Transfer to GPU
        batch_in, batch_out = batch_in.to(device), batch_out.to(device)
        
        # Clear optimizer gradients
        optimizer.zero_grad()
        # Forward pass through the model
        outputs = model(batch_in)
        
        # Calculate loss
        recon_loss = lam_zq*loss_function(outputs['x_out_zq'], batch_out) + lam_ze*loss_function(outputs['x_out_ze'], batch_out)
        if 'vq_loss' in outputs:
            loss = recon_loss + outputs['vq_loss']
        else:
            loss = recon_loss

        # Back propagate
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        batch_run_time = time.time() - batch_run_time
        et = batch_run_time * (len(training_generator) - batch_id) * 3
        # Print progress
        if batch_id % log_every_num_batches == 0 or batch_id == 1:
            print('Train Epoch: {:d} [{:d}/{:d} ({:.0f}%)]\tLoss: {:.6f}\tET - {:s}'.format(
                    epoch+1, batch_id, len(training_generator),
                    100. * batch_id / len(training_generator),
                    loss.item() / len(batch_in), str(datetime.timedelta(seconds=int(et)))))
        batch_id += 1
        
    t_epoch = time.time() - t0
    train_loss /= len(training_generator) * batch_size
    print()
    print('====> Epoch: {} Average loss: {:.4f}\tTime elapsed: {:s}'.format(
          epoch+1, train_loss, str(datetime.timedelta(seconds=int(t_epoch)))))
    return train_loss



"""
Evaluates a VAE model with a given validation/test set

This is a modified form of the test() function meant to be used on models that have 
both the reconstruction outputs from the encoder and quantized layer. 

Arguments
    - model: torch.nn.module, in this case it's the VAE3D class from the Models.py file
    - loss_function: pointer to a function to compute model loss, should take in arguments 
      (original image, reconstructed image, latent mean, latent logvariance)
    - validation_generator: pytorch DataLoader object that is a generator to yield batches of test data
"""
def test_NewVQVAE(model, loss_function, validation_generator):
    # Setup GPU if available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    if use_cuda:
        if torch.cuda.device_count() > 1:
          print("Let's use", torch.cuda.device_count(), "GPUs!")
          # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
          model = nn.DataParallel(model)
    model.to(device)

    model.eval()
    test_loss = 0
    t0 = time.time()
    batch_size = validation_generator.batch_size
    with torch.no_grad():
        for batch_in, batch_out in validation_generator:
            # Transfer to GPU
            batch_in, batch_out = batch_in.to(device), batch_out.to(device)
            # Forward pass through the model
            outputs = model(batch_in)
            
            # Calculate loss
            recon_loss = loss_function(outputs['x_out_ze'], batch_out)
            if 'vq_loss' in outputs:
                loss = recon_loss + outputs['vq_loss']
            else:
                loss = recon_loss
            test_loss += loss.item()
    
    t_epoch = time.time() - t0
    test_loss /= len(validation_generator) * batch_size
    print('====> Test set loss: {:.4f}\tTime elapsed: {:s}'.format(
        test_loss, str(datetime.timedelta(seconds=int(t_epoch)))))
    print()
    return test_loss

