import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F

best_score = 0.0


def linear_motion_loss(outputs, mask) -> torch.Tensor:
    #batch_size = outputs.shape[0]
    s_len = outputs.shape[1]

    loss = outputs.new_zeros(1)
    for idx in range(2, s_len, 1):
        # mask loss to valid outputs
        # motion_mask: (B, 1), the mask of current frame
        motion_mask = mask[:, idx].view(mask.shape[0], 1)

        # Loss: |(loc_t - loc_t-1), (loc_t-1, loc_t-2)|_1 for t = [2, s_len]
        # If loc_t is empty, mask it out by motion_mask
        curr_motion = (outputs[:, idx] - outputs[:, idx - 1]) * motion_mask
        past_motion = (outputs[:, idx - 1] - outputs[:, idx - 2]) * motion_mask
        loss += F.l1_loss(past_motion, curr_motion)
    return loss / (torch.sum(mask))


def freeze_model(model):
    for m in model.modules():
        m.eval()
        for p in m.parameters():
            p.requires_grad = False


def freeze_bn(model, freeze_bn_running=True, freeze_bn_affine=False):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if freeze_bn_running:
                m.eval()  # Freezing Mean/Var of BatchNorm2D.
            if freeze_bn_affine:
                # Freezing Weight/Bias of BatchNorm2D.
                for p in m.parameters():
                    p.requires_grad = False


def load_checkpoint(model, ckpt_path, optimizer=None, is_test=False):
    global best_score
    assert os.path.isfile(ckpt_path), (
        "No checkpoint found at '{}'".format(ckpt_path))
    print("=> Loading checkpoint '{}'".format(ckpt_path))
    checkpoint = torch.load(ckpt_path)
    if 'best_score' in checkpoint:
        best_score = checkpoint['best_score']
    if 'optimizer' in checkpoint and optimizer is not None:
        print("=> Loading optimizer state")
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except (ValueError) as ke:
            print("Cannot load full model: {}".format(ke))
            if is_test: raise ke

    state = model.state_dict()
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except (RuntimeError, KeyError) as ke:
        print("Cannot load full model: {}".format(ke))
        if is_test: raise ke
        state.update(checkpoint['state_dict'])
        model.load_state_dict(state)
    print("=> Successfully loaded checkpoint '{}' (epoch {})".format(
        ckpt_path, checkpoint['epoch']))
    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(state, ckpt_path, check_freq):
    global best_score
    torch.save(state, ckpt_path)

    if state['best_score'] > best_score and state['phase'] == 'val':
        best_score = state['best_score']
        best_path = ckpt_path.replace('latest', 'best')
        shutil.copyfile(ckpt_path, best_path)
    if state['epoch'] % check_freq == 0:
        history_path = ckpt_path.replace('latest',
                                         '{:03d}'.format(state['epoch']))
        shutil.copyfile(ckpt_path, history_path)


def adjust_learning_rate(optimizer,
                         epoch,
                         init_lr,
                         step_ratio: float = 0.5,
                         lr_step: int = 100,
                         lr_adjust: str = 'step',
                         verbose: bool = False):
    if lr_adjust == 'step':
        """Sets the learning rate to the initial LR decayed by 10
        every 30 epochs"""
        lr = init_lr * (step_ratio**(epoch // lr_step))
    else:
        raise ValueError()
    if verbose:
        print('Epoch [{}] Learning rate: {:0.6f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def init_module(layer):
    '''
    Initial modules weights and biases
    '''
    for m in layer.modules():
        if isinstance(m, nn.Conv2d) or \
            isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d) or \
            isinstance(m, nn.GroupNorm):
            m.weight.data.uniform_()
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param.data)
                else:
                    nn.init.normal_(param.data)


def init_lstm_module(layer):
    '''
    Initial LSTM weights and biases
    '''
    for name, param in layer.named_parameters():

        if 'weight_ih' in name:
            torch.nn.init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:
            torch.nn.init.orthogonal_(param.data)
        elif 'bias' in name:
            param.data.fill_(0)  # initializing the lstm bias with zeros
