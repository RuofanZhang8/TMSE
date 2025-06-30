import os
from os.path import join as j_
import pdb
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
import numpy as np
import torch
import torch.nn as nn
from sksurv.metrics import concordance_index_censored
from sksurv.metrics import cumulative_dynamic_auc
from mil_models.tokenizer import PrototypeTokenizer
from mil_models import create_multimodal_survival_model, prepare_emb
from utils.losses import NLLSurvLoss, CoxLoss, SurvRankingLoss
from utils.utils import (EarlyStopping, save_checkpoint, AverageMeter, safe_list_to,
                         get_optim, print_network, get_lr_scheduler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train(datasets, args):
    """
    Train for a single fold for suvival
    """
    
    index_list=[] #record the validation auc/c-index for each epoch
    index_tauc_list=[] #record the validation auc for each epoch
    train_loss_list=[]
    dumps_list=[]
    
    writer_dir = args.results_dir
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    assert args.es_metric == 'loss'
    
    if args.loss_fn == 'nll':
        loss_fn = NLLSurvLoss(alpha=args.nll_alpha)
    elif args.loss_fn == 'cox':
        loss_fn = CoxLoss()
    elif args.loss_fn == 'rank':
        loss_fn = SurvRankingLoss()

    args.feat_dim = args.in_dim # Patch feature dimension
    print('\nInit Model...', end=' ')

    # If prototype-based models, need to create slide-level embeddings
    if args.model_histo_type == 'Gaussian':
        datasets, _ = prepare_emb(datasets, args, mode='survival')

        new_in_dim = None
        for k, loader in datasets.items():
            assert loader.dataset.X is not None
            new_in_dim_curr = loader.dataset.X.shape[-1]
            if new_in_dim is None:
                new_in_dim = new_in_dim_curr
            else:
                assert new_in_dim == new_in_dim_curr

            # The original embedding is 1-D (long) feature vector
            # Reshape it to (n_proto, -1)
            tokenizer = PrototypeTokenizer(args.model_histo_type, args.out_type, args.n_proto)
            prob, mean, cov = tokenizer(loader.dataset.X)
            loader.dataset.X = torch.cat([torch.Tensor(prob).unsqueeze(dim=-1), torch.Tensor(mean), torch.Tensor(cov)], dim=-1)

            factor = args.n_proto
            
        args.in_dim = new_in_dim // factor
    else:
        print(f"{args.model_histo_type} doesn't construct unsupervised slide-level embeddings!")

    ## Set the dimensionality for different inputs
    args.omic_dim = datasets['train'].dataset.omics_data.shape[1]

    if args.omics_modality in ['pathway', 'functional']:
        omic_sizes = datasets['train'].dataset.omic_sizes
    else:
        omic_sizes = []

    model = create_multimodal_survival_model(args, omic_sizes=omic_sizes)
    model.to(device)
    
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model=model, args=args)
    lr_scheduler = get_lr_scheduler(args, optimizer, datasets['train'])

    if args.early_stopping:
        print('\nSetup EarlyStopping...', end=' ')
        early_stopper = EarlyStopping(save_dir=args.results_dir,
                                      patience=args.es_patience,
                                      min_stop_epoch=args.es_min_epochs,
                                      better='min' if args.es_metric == 'loss' else 'max',
                                      verbose=True)
    else:
        print('\nNo EarlyStopping...', end=' ')
        early_stopper = None
    
    #####################
    # The training loop #
    #####################
    for epoch in range(args.max_epochs):
        step_log = {'epoch': epoch, 'samples_seen': (epoch + 1) * len(datasets['train'].dataset)}

        ### Train Loop
        print('#' * 10, f'TRAIN Epoch: {epoch}', '#' * 10)
        train_results = train_loop_survival(model, datasets['train'], optimizer, lr_scheduler, loss_fn,
                                            print_every=args.print_every, accum_steps=args.accum_steps,args=args)


        ### Validation Loop (Optional)
        if 'val' in datasets.keys():
            print('#' * 11, f'VAL Epoch: {epoch}', '#' * 11)
            val_results, dump_epoch  = validate_survival(model, datasets['val'], loss_fn,dump_results=True, 
                                                   print_every=args.print_every, verbose=True,args=args)
            index_list.append(val_results['c_index'])
            index_tauc_list.append(val_results['tAUC'])
            dumps_list.append(dump_epoch)
            
            ### Check Early Stopping (Optional)
            if early_stopper is not None:
                if args.es_metric == 'loss':
                    score = val_results['loss']

                else:
                    raise NotImplementedError
                save_ckpt_kwargs = dict(config=vars(args),
                                        epoch=epoch,
                                        model=model,
                                        score=score,
                                        fname=f's_checkpoint.pth')
                stop = early_stopper(epoch, score, save_checkpoint, save_ckpt_kwargs)
                if stop:
                    break
        print('#' * (22 + len(f'TRAIN Epoch: {epoch}')), '\n')

    ### End of epoch: Load in the best model (or save the latest model with not early stopping)
    if args.early_stopping:
        model.load_state_dict(torch.load(j_(args.results_dir, f"s_checkpoint.pth"))['model'])
    else:
        torch.save(model.state_dict(), j_(args.results_dir, f"s_checkpoint.pth"))

    ### End of epoch: Evaluate on val and test set
    results, dumps = {}, {}
        
    if 'val' in datasets.keys():  
        result_cindex = {'max_val_index': max(index_list), 'max_tauc_cindex':index_tauc_list[np.argmax(index_list)],'max_tauc':max(index_tauc_list),
                     'index_end': index_list[-1], 'index_list': index_list,'index_tauc_list':index_tauc_list,'dumps_list':dumps_list}

        print(f'Validation Index: {index_list}')
        print(f'MAX Validation Index: {max(index_list)}')
        print(f'max_tauc_cindex: {index_tauc_list[np.argmax(index_list)]}')
    return results, dumps, result_cindex


from .MI_utils import calculate_MI
def train_loop_survival(model, loader, optimizer, lr_scheduler, loss_fn=None, 
                        print_every=50, accum_steps=32,args=None):
    
    model.train()
    meters = {'bag_size': AverageMeter()}
    bag_size_meter = meters['bag_size']
    all_risk_scores, all_censorships, all_event_times = [], [], []
    
    x1_tmp = torch.zeros((1, 256)).to(device)
    x2_tmp = torch.zeros((1, 256)).to(device)
    z1_tmp = torch.zeros((1, 256)).to(device)
    z2_tmp = torch.zeros((1, 256)).to(device)
    
    for batch_idx, batch in enumerate(loader):
        data = safe_list_to(batch['img'], device)
        label = safe_list_to(batch['label'], device)

        event_time = batch['survival_time'].to(device)
        censorship = batch['censorship'].to(device)
        attn_mask = batch['attn_mask'].to(device) if ('attn_mask' in batch) else None

        omics = safe_list_to(batch['omics'], device)

        text =safe_list_to(batch['text_emb'], device)
        
        feat = safe_list_to(batch['all_features'], device)

        out, log_dict = model(data, omics, text, feat,attn_mask=attn_mask, label=label, censorship=censorship, loss_fn=loss_fn)

        if out['loss'] is None:
            continue
        
        loss = out['loss']
        #IB loss
        x1, x2, z1, z2 = out['x1'], out['x2'], out['z1'], out['z2']
        x1,x2 = torch.mean(x1, dim=1),torch.mean(x2, dim=1)
        
        if(int((batch_idx+1)/32) == 0):
            x1_tmp = torch.cat((x1_tmp, x1.detach()), dim=0)  
            x2_tmp = torch.cat((x2_tmp, x2.detach()), dim=0)          
            z1_tmp = torch.cat((z1_tmp, z1.detach()), dim=0)          
            z2_tmp = torch.cat((z2_tmp, z2.detach()), dim=0)           

        else:
            x1_tmp = torch.from_numpy(np.delete(x1_tmp.cpu().detach().numpy(), 0, axis=0)).to(device)
            x2_tmp = torch.from_numpy(np.delete(x2_tmp.cpu().detach().numpy(), 0, axis=0)).to(device)
            z1_tmp = torch.from_numpy(np.delete(z1_tmp.cpu().detach().numpy(), 0, axis=0)).to(device)
            z2_tmp = torch.from_numpy(np.delete(z2_tmp.cpu().detach().numpy(), 0, axis=0)).to(device)

            x1_tmp = torch.cat((x1_tmp, x1), dim=0)
            x2_tmp = torch.cat((x2_tmp, x2), dim=0)
            z1_tmp = torch.cat((z1_tmp, z1), dim=0)
            z2_tmp = torch.cat((z2_tmp, z2), dim=0)
            with torch.no_grad():
                x1_numpy = x1_tmp.cpu().detach().numpy()
                k_x1 = squareform(pdist(x1_numpy, 'euclidean'))
                sigma_x1 = np.mean(np.mean(np.sort(k_x1[:, :10], 1)))

                x2_numpy = x2_tmp.cpu().detach().numpy()
                k_x2 = squareform(pdist(x2_numpy, 'euclidean'))
                sigma_x2 = np.mean(np.mean(np.sort(k_x2[:, :10], 1)))

                Z1_numpy = z1_tmp.cpu().detach().numpy()
                Z2_numpy = z2_tmp.cpu().detach().numpy()
                k_z1 = squareform(pdist(Z1_numpy, 'euclidean'))
                k_z2 = squareform(pdist(Z2_numpy, 'euclidean'))
                sigma_z1 = np.mean(np.mean(np.sort(k_z1[:, :10], 1)))
                sigma_z2 = np.mean(np.mean(np.sort(k_z2[:, :10], 1)))
            
            IX1Z1 = calculate_MI(x1_tmp, z1_tmp, s_x=sigma_x1 ** 2, s_y=sigma_z1 ** 2)
            IX2Z2 = calculate_MI(x2_tmp, z2_tmp, s_x=sigma_x2 ** 2, s_y=sigma_z2 ** 2)
            beta1 = 0.01  
            beta2 = 0.01 
            MIB_loss = beta1 * IX1Z1 + beta2 * IX2Z2
            loss = loss + MIB_loss

        
        loss = out['loss']
        loss = loss / accum_steps
        loss.backward()
        if (batch_idx + 1) % accum_steps == 0:
            optimizer.step()
            if args.model_mm_type =='mhim':
                model.update_teacher()
            lr_scheduler.step()
            optimizer.zero_grad()

        # End of iteration survival-specific metrics to calculate / log
        all_risk_scores.append(out['risk'].detach().cpu().numpy())
        all_censorships.append(censorship.cpu().numpy())
        all_event_times.append(event_time.cpu().numpy())

        for key, val in log_dict.items():
            if key not in meters:
                meters[key] = AverageMeter()
            meters[key].update(val, n=len(data))

        bag_size_meter.update(data.size(1), n=len(data))

        if ((batch_idx + 1) % print_every == 0) or (batch_idx == len(loader) - 1):
            msg = [f"avg_{k}: {meter.avg:.4f}" for k, meter in meters.items()]
            msg = f"batch {batch_idx}\t" + "\t".join(msg)
            print(msg)

    # End of epoch survival-specific metrics to calculate / log
    all_risk_scores = np.concatenate(all_risk_scores).squeeze(1)
    all_censorships = np.concatenate(all_censorships).squeeze(1)
    all_event_times = np.concatenate(all_event_times).squeeze(1)
    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    results = {k: meter.avg for k, meter in meters.items()}
    results.update({'c_index': c_index})
    results['lr'] = optimizer.param_groups[0]['lr']
    return results


@torch.no_grad()
def validate_survival(model, loader,
                      loss_fn=None,
                      print_every=50,
                      dump_results=True,
                      recompute_loss_at_end=True,
                      return_attn=False,
                      verbose=1,args=None):
    model.eval()
    meters = {'bag_size': AverageMeter()}
    bag_size_meter = meters['bag_size']
    all_risk_scores, all_censorships, all_event_times = [], [], []
    all_omic_attn, all_cross_attn, all_path_attn = [], [], []

    for batch_idx, batch in enumerate(loader):
        data = batch['img'].to(device)
        label = batch['label'].to(device)
        omics = safe_list_to(batch['omics'], device)

        event_time = batch['survival_time'].to(device)
        censorship = batch['censorship'].to(device)
        attn_mask = batch['attn_mask'].to(device) if ('attn_mask' in batch) else None
        
        text =safe_list_to(batch['text_emb'], device)
        
        feat = safe_list_to(batch['all_features'], device)
        
        out, log_dict = model(data, omics, text,feat, attn_mask=attn_mask, label=label, censorship=censorship, loss_fn=loss_fn, return_attn=return_attn)
        if return_attn:
            all_omic_attn.append(out['omic_attn'].detach().cpu().numpy())
            all_cross_attn.append(out['cross_attn'].detach().cpu().numpy())
            all_path_attn.append(out['path_attn'].detach().cpu().numpy())
        # End of iteration survival-specific metrics to calculate / log
        bag_size_meter.update(data.size(1), n=len(data))
        for key, val in log_dict.items():
            if key not in meters:
                meters[key] = AverageMeter()
            meters[key].update(val, n=len(data))
        all_risk_scores.append(out['risk'].cpu().numpy())
        all_censorships.append(censorship.cpu().numpy())
        all_event_times.append(event_time.cpu().numpy())

        if verbose and (((batch_idx + 1) % print_every == 0) or (batch_idx == len(loader) - 1)):
            msg = [f"avg_{k}: {meter.avg:.4f}" for k, meter in meters.items()]
            msg = f"batch {batch_idx}\t" + "\t".join(msg)
            print(msg)

    

    # End of epoch survival-specific metrics to calculate / log
    all_risk_scores = np.concatenate(all_risk_scores).squeeze(1)
    all_censorships = np.concatenate(all_censorships).squeeze(1)
    all_event_times = np.concatenate(all_event_times).squeeze(1)
    if return_attn:
        if len(all_omic_attn[0].shape) == 2:
            all_omic_attn = np.stack(all_omic_attn)
            all_cross_attn = np.stack(all_cross_attn)
            all_path_attn = np.stack(all_path_attn)
        else:
            all_omic_attn = np.vstack(all_omic_attn)
            all_cross_attn = np.vstack(all_cross_attn)
            all_path_attn = np.vstack(all_path_attn)

    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    #tAUC
    from sksurv.util import Surv
    surv_data = Surv.from_arrays(event=(1 - all_censorships).astype(bool), time=all_event_times)
    unique_event_times = np.unique(all_event_times[all_censorships == 0])
    time_points = np.linspace(unique_event_times.min(), unique_event_times.max()-1e-6, num=1000)
    times,t_auc_values = cumulative_dynamic_auc(surv_data, surv_data, all_risk_scores, time_points)  #return times, t_auc_values  clear nan
    
    results = {k: meter.avg for k, meter in meters.items()}
    results.update({'c_index': c_index,'tAUC':t_auc_values})

    if recompute_loss_at_end and isinstance(loss_fn, CoxLoss):
        surv_loss_dict = loss_fn(logits=torch.tensor(all_risk_scores).unsqueeze(1),
                                 times=torch.tensor(all_event_times).unsqueeze(1),
                                 censorships=torch.tensor(all_censorships).unsqueeze(1))
        results['surv_loss'] = surv_loss_dict['loss'].item()
        results.update({k: v.item() for k, v in surv_loss_dict.items() if isinstance(v, torch.Tensor)})

    if verbose:
        msg = [f"{k}: {v:.3f}" for k, v in results.items()]
        print("\t".join(msg))

    dumps = {}
    if dump_results:
        dumps['all_risk_scores'] = all_risk_scores
        dumps['all_censorships'] = all_censorships
        dumps['all_event_times'] = all_event_times
        dumps['sample_ids'] = np.array(
            loader.dataset.idx2sample_df['sample_id'])
        if return_attn:
            dumps['all_omic_attn'] = all_omic_attn
            dumps['all_cross_attn'] = all_cross_attn
            dumps['all_path_attn'] = all_path_attn
    return results, dumps