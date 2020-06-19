# using: encoding-utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import time
import os
from six.moves import cPickle

import utils.opts as opts
import models
from utils.dataloader import *
import torch.utils.tensorboard as td
import utils.eval_utils as eval_utils
import utils.utils as utils
from utils.rewards import init_cider_scorer, get_self_critical_reward, get_self_critical_cider_bleu_reward, init_bleu_scorer

opt = opts.parse_opt()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

def train(opt):
    opt.use_att = utils.if_use_att(opt.caption_model)
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    td_summary_writer = td.writer.SummaryWriter(opt.ckpt_path)

    infos = {
        'iter': 0,
        'epoch': 0,
        'loader_state_dict': None,
        'vocab': loader.get_vocab(),
    }
    histories = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl'), 'rb') as f:
            infos = cPickle.load(f, encoding='latin-1')
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers", "embed_weight_file"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl'), 'rb') as f:
                histories = cPickle.load(f, encoding='latin-1')


    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    iteration = infos['iter']
    epoch = infos['epoch']
    # For back compatibility
    if 'iterators' in infos:
        infos['loader_state_dict'] = {split: {'index_list': infos['split_ix'][split], 'iter_counter': infos['iterators'][split]} for split in ['train', 'val', 'test']}
    loader.load_state_dict(infos['loader_state_dict'])

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    model = models.setup(opt)
    model.cuda()

    update_lr_flag = True
    # Assure in training mode
    model.train()

    crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()

    optimizer = utils.NewNoamOpt(optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0, betas=(0.9, 0.98), eps=1e-9), max_lr=opt.learning_rate, warmup=opt.newnoamopt_warmup, batchsize=opt.batch_size, decay_start=opt.newnoamopt_decay, datasize=len(loader.dataset.split_ix['train']))
    if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate, betas=(opt.optim_alpha, opt.optim_beta),
                               eps=opt.optim_epsilon, weight_decay=opt.weight_decay)

    params = list(model.named_parameters())
    grad_norm = np.zeros(len(params))
    loss_sum = 0

    while True:
        if opt.self_critical_after != -1 and epoch >= opt.self_critical_after and update_lr_flag and opt.caption_model in ['svbase' ,'umv']:
            print('start self critical')
            if epoch >= 15 and epoch <20 and opt.learning_rate_decay_start >= 0:
                opt.current_lr = opt.learning_rate
            elif epoch >= 20 and opt.learning_rate_decay_start >= 0:
                opt.current_lr = opt.learning_rate / 2.0
            utils.set_lr(optimizer, opt.current_lr)
            update_lr_flag = False
            # Assign the scheduled sampling prob
        if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
            frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
            opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
            model.ss_prob = opt.ss_prob


        # If start self critical training
        if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
            sc_flag = True
            opt.embed_weight_requires_grad = True
            init_cider_scorer(opt.cached_tokens)
            init_bleu_scorer()
        else:
            sc_flag = False
            opt.embed_weight_requires_grad = False


        start = time.time()
        # Load data from train split (0)
        data = loader.get_batch('train')
        print('Read data:', time.time() - start)

        torch.cuda.synchronize()
        start = time.time()

        num_bbox, att_feats = data['num_bbox'].cuda(), data['att_feats'].cuda()
        labels = data['labels'].cuda()
        masks = data['masks'].cuda()
        
        optimizer.zero_grad()
        if not sc_flag:
            loss = crit(model(att_feats, num_bbox, labels), labels[:, 1:], masks[:,1:])
        else:
            gen_result, sample_logprobs = model.sample(att_feats, num_bbox, opt={'sample_max':0})
            reward = get_self_critical_reward(model, att_feats, num_bbox, data, gen_result)
            loss = rl_crit(sample_logprobs, gen_result, torch.from_numpy(reward).float().cuda())

        loss.backward()
        utils.clip_gradient(optimizer, opt.grad_clip)

        for grad_wt in range(len(params)):
            norm_v = torch.norm(params[grad_wt][1].grad).cpu().data.numpy() if params[grad_wt][
                                                                                   1].grad is not None else 0
            grad_norm[grad_wt] += norm_v
            
        if not sc_flag:
            optimizer.step(epoch)
        else:
            optimizer.step()
        train_loss = loss.item()

        loss_sum += train_loss

        torch.cuda.synchronize()
        end = time.time()
        if not sc_flag:
            print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                .format(iteration, epoch, train_loss, end - start))
        else:
            print("lr {} iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}" \
                .format(opt.current_lr, iteration, epoch, np.mean(reward[:,0]), end - start))

        # Update the iteration and epoch
        iteration += 1
        if sc_flag:
            del gen_result
            del sample_logprobs

        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            if opt.noamopt:
                opt.current_lr = optimizer.rate()
            elif not sc_flag:
                opt.current_lr = optimizer.rate(epoch)
            if td is not None:
                td_summary_writer.add_scalar('train_loss', train_loss, iteration)
                td_summary_writer.add_scalar('learning_rate', opt.current_lr, iteration)
                td_summary_writer.add_scalar('scheduled_sampling_prob', model.ss_prob, iteration)
                if sc_flag:
                    td_summary_writer.add_scalar('avg_reward', np.mean(reward[:,0]), iteration)
                # tf_summary_writer.flush()

            loss_history[iteration] = train_loss if not sc_flag else np.mean(reward[:,0])
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0):
            # eval model
            eval_kwargs = {'split': 'val',
                            'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = eval_utils.eval_split(model, loader, eval_kwargs)

            # Write validation result into summary
            if td is not None:
                td_summary_writer.add_scalar('validation loss', val_loss, iteration)
                if lang_stats is not None:
                    for k,v in lang_stats.items():
                        td_summary_writer.add_scalar(k, v, iteration)
                # tf_summary_writer.flush()
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss

            best_flag = False
            if True: # if true
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                checkpoint_path = os.path.join(opt.ckpt_path, 'model.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(opt.ckpt_path, 'optimizer.pth')
                torch.save(optimizer.state_dict(), optimizer_path)

                # Dump miscalleous informations
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iters
                infos['split_ix'] = loader..dataset.split_ix
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['vocab'] = loader.get_vocab()

                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history
                with open(os.path.join(opt.ckpt_path, 'infos_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(opt.ckpt_path, 'histories_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(histories, f)

                if best_flag:
                    checkpoint_path = os.path.join(opt.ckpt_path, 'model-best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    with open(os.path.join(opt.ckpt_path, 'infos_'+opt.id+'-best.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)
                loss_sum = 0
                grad_norm = np.zeros(len(params))

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            eval_kwargs = {'split': 'val',
                            'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = eval_utils.eval_split(model, loader, eval_kwargs)

            # Write validation result into summary
            if td is not None:
                td_summary_writer.add_scalar('validation loss', val_loss, iteration)
                if lang_stats is not None:
                    for k,v in lang_stats.items():
                        td_summary_writer.add_scalar(k, v, iteration)

            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss

            best_flag = False
            if True: # if true
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                checkpoint_path = os.path.join(opt.ckpt_path, 'model.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(opt.ckpt_path, 'optimizer.pth')
                torch.save(optimizer.state_dict(), optimizer_path)

                # Dump miscalleous informations
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iterators
                infos['split_ix'] = loader.split_ix
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['vocab'] = loader.get_vocab()

                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history
                with open(os.path.join(opt.ckpt_path, 'infos_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(opt.ckpt_path, 'histories_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(histories, f)

                if best_flag:
                    checkpoint_path = os.path.join(opt.ckpt_path, 'model-best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    with open(os.path.join(opt.ckpt_path, 'infos_'+opt.id+'-best.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)
            break
        if sc_flag:
            del loss
            del reward
            del att_feats
            del num_bbox
            del labels
            del masks
            del data

opt = opts.parse_opt()
train(opt)
