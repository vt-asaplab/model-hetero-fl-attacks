import argparse
import copy
import datetime
import models
import numpy as np
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
from torchmetrics.regression import PearsonCorrCoef
from config import cfg
from data import fetch_dataset, make_data_loader, split_dataset, SplitDataset
from fed import Federation
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import Logger
from collections import OrderedDict
import math
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
for k in cfg:
    cfg[k] = args[k]
if args['control_name']:
    cfg['control'] = {k: v for k, v in zip(cfg['control'].keys(), args['control_name'].split('_'))} \
        if args['control_name'] != 'None' else {}
cfg['control_name'] = '_'.join([cfg['control'][k] for k in cfg['control']])
cfg['pivot_metric'] = 'Global-Accuracy'
cfg['pivot'] = -float('inf')
cfg['metric_name'] = {'train': {'Local': ['Local-Loss', 'Local-Accuracy']},
                      'test': {'Local': ['Local-Loss', 'Local-Accuracy'], 'Global': ['Global-Loss', 'Global-Accuracy']}}
cfg['file_output'] = "New_Tables/MNIST_ConvRate_TEST"
cfg['local_train_size'] = 5
full_path = os.getcwd() + "/" + cfg['file_output']
fp = open(full_path, 'w')
fp.write("N Max_Pearson Max_PSNR\n")

def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['subset'], cfg['model_name'], cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        print('---------------------------------------')
        runExperiment()
    return


def runExperiment():
    seed = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dataset = fetch_dataset(cfg['data_name'], cfg['subset'])
    process_dataset(dataset)
    model = eval('models.{}(model_rate=cfg["global_model_rate"]).to(cfg["device"])'.format(cfg['model_name']))
    optimizer = make_optimizer(model, cfg['lr'])
    scheduler = make_scheduler(optimizer)
    if cfg['resume_mode'] == 1:
        last_epoch, data_split, label_split, model, optimizer, scheduler, logger = resume(model, cfg['model_tag'],
                                                                                          optimizer, scheduler)
    elif cfg['resume_mode'] == 2:
        last_epoch = 1
        _, data_split, label_split, model, _, _, _ = resume(model, cfg['model_tag'])
        logger_path = os.path.join('output', 'runs', '{}'.format(cfg['model_tag']))
        logger = Logger(logger_path)
    else:
        last_epoch = 1
        data_split, label_split = split_dataset(dataset, cfg['num_users'], cfg['data_split_mode'])
        logger_path = os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag']))
        logger = Logger(logger_path)
    if data_split is None:
        data_split, label_split = split_dataset(dataset, cfg['num_users'], cfg['data_split_mode'])
    global_parameters = model.state_dict()

    model_history = {}
    model_history['blocks.0.bias'] = []
    model_history['blocks.2.weight'] = []
    model_history['blocks.2.bias'] = []

    model_history_fcnn = {}
    model_history_fcnn['layers.0.weight'] = []
    model_history_fcnn['layers.0.bias'] = []

    model_dist = OrderedDict()
    for epoch in range(last_epoch, cfg['num_epochs']['global'] + 1):
        logger.safe(True)
        federation = Federation(epoch, model_dist, global_parameters, cfg['model_rate'], label_split)
        train(model_history, model_history_fcnn, dataset['train'], data_split['train'], label_split, federation, model, optimizer, logger, epoch)
        model_dist = copy.deepcopy(federation.model_to_distribute)
        test_model = stats(dataset['train'], model)
        test(dataset['test'], data_split['test'], label_split, test_model, logger, epoch)
        if cfg['scheduler_name'] == 'ReduceLROnPlateau':
            scheduler.step(metrics=logger.mean['train/{}'.format(cfg['pivot_metric'])])
        else:
            scheduler.step()
        logger.safe(False)
        model_state_dict = model.state_dict()
        save_result = {
            'cfg': cfg, 'epoch': epoch + 1, 'data_split': data_split, 'label_split': label_split,
            'model_dict': model_state_dict, 'optimizer_dict': optimizer.state_dict(),
            'scheduler_dict': scheduler.state_dict(), 'logger': logger}
        save(save_result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if cfg['pivot'] < logger.mean['test/{}'.format(cfg['pivot_metric'])]:
            cfg['pivot'] = logger.mean['test/{}'.format(cfg['pivot_metric'])]
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    logger.safe(False)
    # fp.close()
    return


def train(model_history, model_history_fcnn, dataset, data_split, label_split, federation, global_model, optimizer, logger, epoch):
    global_model.load_state_dict(federation.global_parameters)
    global_model.train(True)
    local, local_parameters, user_idx, param_idx = make_local(dataset, data_split, label_split, federation)
    num_active_users = len(local)
    
    target_users = []
    converged_users = []
    for m in range(num_active_users):
        if federation.model_rate[user_idx[m]] == 0.5:
            target_users.append(user_idx[m])
        if federation.model_rate[user_idx[m]] == 1:
            converged_users.append(user_idx[m])
    num_updaters = len(target_users) + len(converged_users)

    lr = optimizer.param_groups[0]['lr']

    start_time = time.time()

    local_params_orig = {}
    model_params = {}
    img_list = None
    for m in range(num_active_users):
        lr = cfg['lr_map'][federation.model_rate[user_idx[m]]] # This line modifies the learning rate based on user. 

        (local_params_orig[m], img_data) = local[m].train(local_parameters[m], lr, logger)
        local_parameters[m] = copy.deepcopy(local_params_orig[m])
        img_list = copy.deepcopy(img_data)

        if (user_idx[m] in converged_users):
            weight_grad = torch.subtract(global_model.state_dict()['layers.0.weight'], local_parameters[m]['layers.0.weight'])
            bias_grad = torch.subtract(global_model.state_dict()['layers.0.bias'], local_parameters[m]['layers.0.bias'])
        if (user_idx[m] in target_users and epoch == cfg['num_epochs']['global']):
            weight_grad = torch.subtract(federation.model_to_distribute['layers.0.weight'].to(cfg["device"]), local_parameters[m]['layers.0.weight'])
            bias_grad = torch.subtract(federation.model_to_distribute['layers.0.bias'].to(cfg["device"]), local_parameters[m]['layers.0.bias'])

        if m % int((num_active_users * cfg['log_interval']) + 1) == 0:
            local_time = (time.time() - start_time) / (m + 1)
            epoch_finished_time = datetime.timedelta(seconds=local_time * (num_active_users - m - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['num_epochs']['global'] - epoch) * local_time * num_active_users))
            info = {'info': ['Model: {}'.format(cfg['model_tag']), 
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * m / num_active_users),
                             'ID: {}({}/{})'.format(user_idx[m], m + 1, num_active_users),
                             'Learning rate: {}'.format(lr),
                             'Rate: {}'.format(federation.model_rate[user_idx[m]]),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            logger.write('train', cfg['metric_name']['train']['Local'])    
        
        if cfg['model_name'] == 'conv':
            if user_idx[m] in converged_users:
                for k, v in local_parameters[m].items():
                    local_grad_maxpool_weight = torch.subtract(local_parameters[m]['blocks.2.weight'], global_model.state_dict()['blocks.2.weight'])
                    local_grad_maxpool_bias = torch.subtract(local_parameters[m]['blocks.2.bias'], global_model.state_dict()['blocks.2.bias'])
                    print("Local Gradient Maxpool Weight For key %s for user %s = %s"%(k, user_idx[m], local_grad_maxpool_weight), flush=True)
                    print("Local Gradient Maxpool Bias For key %s for user %s = %s"%(k, user_idx[m], local_grad_maxpool_bias), flush=True)

                    weight_over_bias = torch.div(local_grad_maxpool_weight, local_grad_maxpool_bias)
                    print(len(weight_over_bias))


        
    
    federation.combine(local_parameters, param_idx, user_idx)
    global_model.load_state_dict(federation.global_parameters)


    global_model_state_dict_copy = copy.deepcopy(global_model.state_dict())

    # Append to the model history. 
    if cfg['model_name'] == 'fcnn':
        # W_{j+1} = (A1 + B1) / 2. 
        # W_j = (A0 + B0) / 2. 
        # W_{j+1} - W_j = (grad_A + grad_B) / 2. 
        # grad_A approx. 0, so W_{j+1} - W_{j} = grad_B / 2. 
        # Compare with actual grad_B (using local_params and model_to_distribute). 
        # Then, using grad_B for weight and bias, leak training samples. 
        model_history_fcnn['layers.0.weight'].append(global_model_state_dict_copy['layers.0.weight'])
        model_history_fcnn['layers.0.bias'].append(global_model_state_dict_copy['layers.0.bias'])

        for m in range(num_active_users):
            if user_idx[m] in target_users:
                weight_grad = None
                bias_grad = None
                max_pearson_overall = 0.0
                max_psnr_overall = 0.0
                max_pearson_list = []
                max_psnr_list = []
                for k, v in global_model.state_dict().items():
                    if epoch == cfg['num_epochs']['global']: 
                        if k == 'layers.0.weight': 
                            # Call fcnn_leakage. 
                            weight_grad = fcnn_leakage(epoch, k, user_idx[m], num_active_users, federation.model_rate[user_idx[m]], local_parameters[m][k], model_history_fcnn[k], federation.model_to_distribute[k])
                        elif k == 'layers.0.bias':
                            bias_grad = fcnn_leakage(epoch, k, user_idx[m], num_active_users, federation.model_rate[user_idx[m]], local_parameters[m][k], model_history_fcnn[k], federation.model_to_distribute[k])
                    
                    if (weight_grad is not None and bias_grad is not None):
                        bias_grad_sum = torch.abs(torch.sum(bias_grad)).item()
                        if bias_grad_sum != 0.0:
                            (max_pearson, max_psnr) = reconstruct_image(weight_grad, bias_grad, img_list)
                            max_pearson_overall = max(max_pearson_overall, max_pearson)
                            max_psnr_overall = max(max_psnr_overall, max_psnr)
                            max_pearson_list.append(max_pearson)
                            max_psnr_list.append(max_psnr)
                if len(max_pearson_list) > 0:
                    N_Table = cfg['local_train_size']
                    Max_Pearson_Table = max(max_pearson_list)
                    Avg_Pearson_Table = sum(max_pearson_list) / len(max_pearson_list)
                    Max_PSNR_Table = max(max_psnr_list)
                    Avg_PSNR_Table = sum(max_psnr_list) / len(max_psnr_list)
                    fp.write("%s %s %s\n"%(N_Table, Max_Pearson_Table, Max_PSNR_Table))

    if cfg['model_name'] == 'conv':
        model_history['blocks.0.bias'].append(global_model_state_dict_copy['blocks.0.bias'])
        model_history['blocks.2.weight'].append(global_model_state_dict_copy['blocks.2.weight'])
        model_history['blocks.2.bias'].append(global_model_state_dict_copy['blocks.2.bias'])

    converged_user_gradients = {}
    target_user_gradients = {}
    # Step 1: Compute user gradients.
    if cfg['model_name'] == 'conv': 
        for m in range(num_active_users):
            local_params_size = local_parameters[m]['blocks.0.bias'].size()[0]
            print("TRAIN: local_params_size = %s"%(local_params_size))
            if user_idx[m] in converged_users and epoch >= 2:
                lower = local_params_size // 4
                upper = local_params_size // 2
                print(" TRAIN: local_params for converged_user = %s"%(local_parameters[m]['blocks.0.bias'][lower:upper]))
                print(" TRAIN: converged_user global_model_state_dict_copy['blocks.0.bias'][lower:upper] = %s"%(model_history['blocks.0.bias'][epoch-2][lower:upper]))
                converged_user_gradients['blocks.0.bias'] = local_parameters[m]['blocks.0.bias'][lower:upper] - model_history['blocks.0.bias'][epoch-2][lower:upper]
                print(" TRAIN: converged_user_gradients = %s"%(converged_user_gradients['blocks.0.bias']))
            elif user_idx[m] in target_users and epoch >= 2:
                lower = local_params_size // 2
                upper = local_params_size
                print(" TRAIN: local_params for target_user = %s"%(local_parameters[m]['blocks.0.bias'][lower:upper]))
                print(" TRAIN: target_user global_model_state_dict_copy['blocks.0.bias'][lower:upper] = %s"%(model_history['blocks.0.bias'][epoch-2][lower:upper]))
                target_user_gradients['blocks.0.bias'] = local_parameters[m]['blocks.0.bias'][lower:upper] - model_history['blocks.0.bias'][epoch-2][lower:upper]
                print(" TRAIN: target_user_gradients = %s"%(target_user_gradients['blocks.0.bias']))

    # Step 2: Compute global model. 
    if cfg['model_name'] == 'conv':
        for m in range(num_active_users):
            if user_idx[m] in target_users:
                for k, v in global_model.state_dict().items():
                    if k == 'blocks.0.bias' and epoch >= 2:
                        # Grab the previous global model. 
                        local_params_size = local_parameters[m]['blocks.0.bias'].size()[0]
                        lower = local_params_size // 2
                        upper = local_params_size
                        global_model_with_gradients = copy.deepcopy(model_history['blocks.0.bias'][epoch - 2])
                        global_model_with_gradients_scaled = global_model_with_gradients[lower:upper]
                        global_model_with_gradients_scaled += converged_user_gradients['blocks.0.bias']
                        global_model_with_gradients_scaled += target_user_gradients['blocks.0.bias'] 
                        print(" TRAIN: global_model_with_gradients_scaled = %s"%(global_model_with_gradients_scaled))

                        gradient_leakage(epoch, k, user_idx[m], num_updaters, local_parameters[m][k], model_history, global_model_with_gradients_scaled, target_user_gradients['blocks.0.bias'])

    return

def reconstruct_image(weight_grad, bias_grad, img_list):
    
    print("RECONSTRUCT: weight_grad.size()[0] = ", weight_grad.size()[0])
    count = 0
    avg_psnr = []
    avg_ssim = []
    avg_pearson = []
    num_recovered = 0
    
    # Define rows and columns for plot. 
    rows = 2
    columns = cfg['local_train_size']
    # fig.add_subplot(rows, columns, 1)
    # (fig, axs) = plt.subplots(rows, columns)
    # for ax in axs.reshape(-1):
    #     ax.grid(False)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
        # ax.set_zticks([])
    # plt.axis('off')
    # plt.grid(False)

    # Define rows and columns for plot. 
    rows = 2
    columns = cfg['local_train_size']
    # fig.add_subplot(rows, columns, 1)
    # (fig, axs) = plt.subplots(rows, columns)
    # for ax in axs.reshape(-1):
    #     ax.grid(False)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
        # ax.set_zticks([])
    # plt.axis('off')
    # plt.grid(False)

    for elem in img_list:
        elem_extracted = elem[0].to(cfg["device"])
        max_pearson = 0.0
        max_ssim = 0.0
        max_psnr = 0.0

        pearson = PearsonCorrCoef().to(cfg["device"])

        count = count + 1

        best_partial_recon = None

        for i in range(weight_grad.size()[0]):
            partial_recon = torch.divide(weight_grad[i], bias_grad[i])

            img_reshaped = elem_extracted.reshape(partial_recon.size())

            # Compute Pearson Similarity: 
            pearson_coef = pearson(img_reshaped, partial_recon).item()

            partial_recon_as_img = partial_recon.reshape(elem_extracted.size())
            elem_extracted_numpy = np.squeeze(elem_extracted.cpu().numpy())
            partial_recon_numpy = np.squeeze(partial_recon_as_img.cpu().numpy())
            # Data range calculation based on normalization in data.py. 
            curr_ssim = ssim(elem_extracted_numpy, partial_recon_numpy, data_range=3.245699448)
            # PSNR metric: 
            curr_psnr = psnr(elem_extracted_numpy, partial_recon_numpy, data_range=3.245699448)

            max_ssim = max(max_ssim, curr_ssim)
            max_psnr = max(max_psnr, curr_psnr)
            max_pearson = max(max_pearson, pearson_coef)
            if max_pearson == pearson_coef:
                best_partial_recon = partial_recon
        print("RECONSTRUCT: Max Pearson for image %s = %s"%(count, max_pearson))
        print("RECONSTRUCT: Best ssim for image %s = %s"%(count, max_ssim))
        print("RECONSTRUCT: Best psnr for image %s = %s"%(count, max_psnr))

        if max_pearson >= 0.98:
            num_recovered = num_recovered + 1

        avg_ssim.append(max_ssim)
        avg_psnr.append(max_psnr)
        avg_pearson.append(max_pearson)

        partial_recon_as_img = best_partial_recon.reshape(elem_extracted.size())

    print("RECONSTRUCT: avg_ssim = %s"%(avg_ssim))
    print("RECONSTRUCT: avg_psnr = %s"%(avg_psnr))
    print("RECONSTRUCT: avg_pearson = %s"%(avg_pearson))

    print("RECONSTRUCT: best_ssim across images = %s"%(max(avg_ssim)))
    print("RECONSTRUCT: best_psnr across images = %s"%(max(avg_psnr)))
    print("RECONSTRUCT: best_pearson across images = %s"%(max(avg_pearson)))
    print("RECONSTRUCT: num_recovered = %s"%(num_recovered))

    print("RECONSTRUCT: avg_ssim across images = %s"%(sum(avg_ssim) / len(avg_ssim)))
    print("RECONSTRUCT: avg_psnr across images = %s"%(sum(avg_psnr) / len(avg_psnr)))
    print("RECONSTRUCT: avg_pearson across images = %s"%(sum(avg_pearson) / len(avg_pearson)))

    # plt.show() 
    # plt.savefig('orig_recon_convrate_n%s_v2.png'%(cfg['local_train_size']))

    return (max(avg_pearson), max(avg_psnr))

def fcnn_leakage(epoch, k, user, num_active_users, model_rate, local_params, model_history, distributed_model):
    local_params_resized = copy.deepcopy(local_params)
    model_history_params_resized = copy.deepcopy(model_history[epoch - 2])
    model_history_params_curr_resized = copy.deepcopy(model_history[epoch - 1])

    lower = local_params.size()[0] // 2
    upper = local_params.size()[0]

    local_params_resized = local_params_resized[lower:upper]
    model_history_params_resized = model_history_params_resized[lower:upper]
    model_history_params_curr_resized = model_history_params_curr_resized[lower:upper]
    
    print(": In fcnn_leakage: model_history_params_resized = %s"%(model_history_params_resized))
    print(": In fcnn_leakage: model_history_params_curr_resized = %s"%(model_history_params_curr_resized))

    # Algebra: 
    # W_{j+1} = (A1 + B1) / 2. 
    # W_j = (A0 + B0) / 2. 
    # W_{j+1} - W_j = (grad_A + grad_B) / 2. 
    # grad_A approx. 0, so W_{j+1} - W_{j} = grad_B / 2. 
    # B1_up = (B1 - W_mal)
    # B0_up = (B0 - W_orig)
    # Compare with actual grad_B (using local_params and model_to_distribute). 
    # Then, using grad_B for weight and bias, leak training samples.
    potential_B = torch.multiply(model_history_params_curr_resized, 2)
    potential_B = torch.subtract(potential_B, model_history_params_resized)
    actual_B = local_params_resized
    print("potential_B = %s and actual_B = %s"%(potential_B, actual_B))
    diff_B = torch.subtract(potential_B, actual_B)
    print(": Diff between potential_B and actual_B = %s"%(diff_B))
    diff_B_abs = torch.abs(diff_B)
    L1_dist = torch.sum(diff_B_abs).item()
    print(": L1 distance between potential and actual B = %s"%(L1_dist))
    user_grad = torch.subtract(distributed_model[lower:upper].to(cfg["device"]), potential_B)
    print(": user_grad = %s"%(user_grad))

    # Concatenate the lower part of the global gradient with user_grad. 
    user_grad_concat = torch.cat((user_grad, user_grad))
    print(" FCNN_LEAKAGE: user_grad_concat = %s"%(user_grad_concat))
    return user_grad_concat


def gradient_leakage(epoch, k, user, num_updaters, local_params, model_history, global_params, target_gradient):
    print(": In gradient_leakage: k = ", k)

    # Step 1: Compute user gradients. 
    # Grad_A = local_params_A - global_params
    # Grad_B = local_params_B - global_params. 

    # Step 2: Compute global model through W_j + Grad_A + Grad_B. 

    # Step 3: Show that when Grad_A converges, server can extract Grad_B, since it knows W_j. 
    # I.e. W_{j+1} = W_j + Grad_A + Grad_B. 
    # So, Grad_B = W_{j+1} - W_j. 
    global_params_resized = copy.deepcopy(global_params)
    local_params_resized = copy.deepcopy(local_params)
    print(": In gradient_leakage: Key = %s and user = %s and epoch = %s and num_updaters = %s"%(k, user, epoch, num_updaters))
    model_history_params_resized = copy.deepcopy(model_history[k][epoch - 2])
    model_history_params_curr_resized = copy.deepcopy(model_history[k][epoch - 1])


    lower = local_params.size()[0] // 2
    upper = local_params.size()[0]

    local_params_resized = local_params_resized[lower:upper]
    model_history_params_resized = model_history_params_resized[lower:upper]
    model_history_params_curr_resized = model_history_params_curr_resized[lower:upper]
    
    print(": In gradient_leakage: model_history_params_resized = %s"%(model_history_params_resized))
    print(": In gradient_leakage: global_params_resized = %s"%(global_params_resized))

    # Algebra: 
    # ((A + B) / 2) - W_j = (A + B - 2 * W_j) / 2 = (B - W_j) / 2 
    potential_grad_B = torch.subtract(model_history_params_curr_resized, model_history_params_resized) # W_{j+1} - W_j. 
    actual_grad_B = torch.div(target_gradient, 2)
    print(": In gradient_leakage: potential_grad_B = %s"%(potential_grad_B))
    print(": In gradient_leakage: actual_grad_B = %s"%(actual_grad_B))
    diff_grad_B = torch.subtract(potential_grad_B, actual_grad_B)
    print(": Diff between potential_grad_B and actual_grad_B = %s"%(diff_grad_B))
    diff_grad_B_abs = torch.abs(diff_grad_B)
    L1_dist = torch.sum(diff_grad_B_abs).item()
    print(": L1 distance between potential and actual grad_B = %s"%(L1_dist))

    global_local_param_diff = torch.subtract(model_history_params_curr_resized, local_params_resized)
    print(": In gradient_leakage: global_local_param_diff = %s"%(global_local_param_diff))
    global_local_param_diff_L1 = torch.abs(global_local_param_diff)

    count = 0
    print(": In gradient_leakage: global_local_param_diff.size()[0] = %s"%(global_local_param_diff.size()[0]))
    for i in range(0, global_local_param_diff.size()[0]):
        print("global_local_param_diff_L1[%s] = %s"%(i, global_local_param_diff_L1[i]))
        if (global_local_param_diff_L1[i] < 0.0001):
            count = count + 1
    print("count for blocks.0.bias epoch %s = %s"%(epoch, count))
    
    # Compute L1 distance
    L1_dist = torch.sum(global_local_param_diff_L1).item()

    # Compute L2 distance. 
    global_local_param_diff_squared = torch.square(global_local_param_diff)
    sum_squared_diff = torch.sum(global_local_param_diff_squared).item()
    L2_dist = math.sqrt(sum_squared_diff)
    
    print(": In gradient_leakage: L1 dist = %s and L2 dist = %s"%(L1_dist, L2_dist))
    if epoch == cfg['num_epochs']['global']:
        print(": In gradient_leakage: epoch = %s before writing to file"%(epoch))
        fp.write("%s %s %s %s\n"%(k, L1_dist, L2_dist, count))
        fp.flush()

    

def stats(dataset, model):
    with torch.no_grad():
        test_model = eval('models.{}(model_rate=cfg["global_model_rate"], track=True).to(cfg["device"])'
                          .format(cfg['model_name']))
        test_model.load_state_dict(model.state_dict(), strict=False)
        data_loader = make_data_loader({'train': dataset})['train']
        test_model.train(True)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input = to_device(input, cfg['device'])
            test_model(input)
    return test_model


def test(dataset, data_split, label_split, model, logger, epoch):
    with torch.no_grad():
        metric = Metric()
        model.train(False)
        for m in range(cfg['num_users']):
            data_loader = make_data_loader({'test': SplitDataset(dataset, data_split[m])})['test']
            for i, input in enumerate(data_loader):
                input = collate(input)
                input_size = input['img'].size(0)
                input['label_split'] = torch.tensor(label_split[m])
                input = to_device(input, cfg['device'])
                output = model(input)
                output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
                evaluation = metric.evaluate(cfg['metric_name']['test']['Local'], input, output)
                logger.append(evaluation, 'test', input_size)
        data_loader = make_data_loader({'test': dataset})['test']
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['img'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input)
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(cfg['metric_name']['test']['Global'], input, output)
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']),
                         'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        logger.write('test', cfg['metric_name']['test']['Local'] + cfg['metric_name']['test']['Global'])
    return


def make_local(dataset, data_split, label_split, federation):
    num_active_users = int(np.ceil(cfg['frac'] * cfg['num_users']))
    user_idx = torch.arange(cfg['num_users'])[torch.randperm(cfg['num_users'])[:num_active_users]].tolist()
    print("MAKE LOCAL: num_active_users = ", num_active_users)
    print("MAKE LOCAL: user_idx = ", user_idx)
    local_parameters, param_idx = federation.distribute(user_idx)
    local = [None for _ in range(num_active_users)]

    for m in range(num_active_users):
        model_rate_m = federation.model_rate[user_idx[m]]
        data_loader_m = make_data_loader({'train': SplitDataset(dataset, data_split[user_idx[m]])})['train']
        local[m] = Local(model_rate_m, data_loader_m, label_split[user_idx[m]])
    return local, local_parameters, user_idx, param_idx


class Local:
    def __init__(self, model_rate, data_loader, label_split):
        self.model_rate = model_rate
        self.data_loader = data_loader
        self.label_split = label_split 

    def train(self, local_parameters, lr, logger):
        metric = Metric()
        model = eval('models.{}(model_rate=self.model_rate).to(cfg["device"])'.format(cfg['model_name']))
        model.load_state_dict(local_parameters)
        model.train(True)
        optimizer = make_optimizer(model, lr)
        input_list = []
        for local_epoch in range(1, cfg['num_epochs']['local'] + 1):
            count = 0
            for i, input in list(enumerate(self.data_loader))[:cfg['local_train_size']]:
                input_list.append(input['img'])
                count = count + 1
                input = collate(input)
                input_size = input['img'].size(0)
                input['label_split'] = torch.tensor(self.label_split)
                input = to_device(input, cfg['device'])
                optimizer.zero_grad()
                output = model(input)
                output['loss'].backward()

                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                optimizer.step()
                evaluation = metric.evaluate(cfg['metric_name']['train']['Local'], input, output)
                logger.append(evaluation, 'train', n=input_size)
            print(" LOCAL TRAIN: count = %s"%(count))
        
        local_parameters = model.state_dict()

        return (local_parameters, input_list)


if __name__ == "__main__":
    main()