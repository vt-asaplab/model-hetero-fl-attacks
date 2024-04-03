import copy
import torch
import numpy as np
from config import cfg
from collections import OrderedDict



class Federation:
    def __init__(self, epoch, global_parameters, rate, label_split):
        self.rd = epoch
        self.global_parameters = global_parameters
        self.target_weights = ['blocks.2.weight']
        self.target_weights_fcnn = ['layers.0.weight']
        self.target_biases = ['blocks.0.bias', 'blocks.2.bias']
        self.target_biases_fcnn = ['layers.0.bias']
        self.rate = rate
        self.label_split = label_split
        self.model_to_distribute = OrderedDict()

        self.make_model_rate()

    def make_model_rate(self):
        if cfg['model_split_mode'] == 'dynamic':
            rate_idx = torch.multinomial(torch.tensor(cfg['proportion']), num_samples=cfg['num_users'],
                                         replacement=True).tolist()
            self.model_rate = np.array(self.rate)[rate_idx]
        elif cfg['model_split_mode'] == 'fix':
            self.model_rate = np.array(self.rate)
        else:
            raise ValueError('Not valid model split mode')
        return

    def split_model(self, user_idx):
        if cfg['model_name'] == 'fcnn':
            idx_i = [None for _ in range(len(user_idx))]
            idx = [OrderedDict() for _ in range(len(user_idx))]
            output_weight_name = [k for k in self.global_parameters.keys() if 'weight' in k][-1]
            output_bias_name = [k for k in self.global_parameters.keys() if 'bias' in k][-1]

            for k, v in self.global_parameters.items():
                parameter_type = k.split('.')[-1]
                for m in range(len(user_idx)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if parameter_type == 'weight':
                            if v.dim() > 1:
                                input_size = v.size(1)
                                output_size = v.size(0)

                                if idx_i[m] is None:
                                    idx_i[m] = torch.arange(input_size, device=v.device)
                                input_idx_i_m = idx_i[m]

                                if k == output_weight_name:
                                    output_idx_i_m = torch.arange(output_size, device=v.device)
                                else:
                                    scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                    local_output_size = int(np.ceil(output_size * scaler_rate))
                                    output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]

                                if k in self.target_weights_fcnn:
                                    if self.model_rate[user_idx[m]] == 0.25 and self.rd == 2:
                                        # Assign this client the malicious weights. 
                                        output_idx_i_m = (output_idx_i_m + int((self.model_rate[user_idx[m]]) * v.size()[0])) % v.size()[0]
                                        (output_idx_i_m, sorted_indeces) = torch.sort(output_idx_i_m)

                                idx[m][k] = output_idx_i_m, input_idx_i_m 
                                idx_i[m] = output_idx_i_m
                            else:
                                input_idx_i_m = idx_i[m]
                                idx[m][k] = input_idx_i_m
                        else:
                            if k == output_bias_name:
                                input_idx_i_m = idx_i[m] # Length of bias tensor should be the same as the length of the weight tensor. 
                                idx[m][k] = input_idx_i_m
                            else:
                                input_idx_i_m = idx_i[m]
                                idx[m][k] = input_idx_i_m
                    else:
                        pass
        elif cfg['model_name'] == 'conv':
            idx_i = [None for _ in range(len(user_idx))]
            idx = [OrderedDict() for _ in range(len(user_idx))]
            output_weight_name = [k for k in self.global_parameters.keys() if 'weight' in k][-1]
            output_bias_name = [k for k in self.global_parameters.keys() if 'bias' in k][-1]
            for k, v in self.global_parameters.items():
                parameter_type = k.split('.')[-1]
                for m in range(len(user_idx)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if parameter_type == 'weight':
                            if v.dim() > 1:
                                input_size = v.size(1)
                                output_size = v.size(0)
                                if idx_i[m] is None:
                                    idx_i[m] = torch.arange(input_size, device=v.device)
                                input_idx_i_m = idx_i[m]
                                if k == output_weight_name:
                                    output_idx_i_m = torch.arange(output_size, device=v.device)
                                else:
                                    scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                    local_output_size = int(np.ceil(output_size * scaler_rate))
                                    output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]
                                idx[m][k] = output_idx_i_m, input_idx_i_m   # Basically just a tensor containing the indeces to update for a given layer. 
                                idx_i[m] = output_idx_i_m  # Update idx_i, because the output of layer j is the input of layer j+1. 
                            else:
                                input_idx_i_m = idx_i[m]

                                if k in self.target_weights:
                                    input_idx_i_m = (input_idx_i_m + (self.rd - 1)) % v.size()[0]
                                    (input_idx_i_m, sorted_indeces) = torch.sort(input_idx_i_m)

                                idx[m][k] = input_idx_i_m
                        else:
                            if k == output_bias_name:
                                input_idx_i_m = idx_i[m] # Length of bias tensor should be the same as the length of the weight tensor. 
                                idx[m][k] = input_idx_i_m
                            else:
                                input_idx_i_m = idx_i[m]

                                if k in self.target_biases:
                                    input_idx_i_m = (input_idx_i_m + (self.rd - 1)) % v.size()[0]
                                    (input_idx_i_m, sorted_indeces) = torch.sort(input_idx_i_m)

                                idx[m][k] = input_idx_i_m
                    else:
                        pass
        elif 'resnet' in cfg['model_name']:
            idx_i = [None for _ in range(len(user_idx))]
            idx = [OrderedDict() for _ in range(len(user_idx))]
            for k, v in self.global_parameters.items():
                parameter_type = k.split('.')[-1]
                for m in range(len(user_idx)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if parameter_type == 'weight':
                            if v.dim() > 1:
                                input_size = v.size(1)
                                output_size = v.size(0)
                                if 'conv1' in k or 'conv2' in k:
                                    if idx_i[m] is None:
                                        idx_i[m] = torch.arange(input_size, device=v.device)
                                    input_idx_i_m = idx_i[m]
                                    scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                    local_output_size = int(np.ceil(output_size * scaler_rate))
                                    output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]
                                    idx_i[m] = output_idx_i_m
                                elif 'shortcut' in k:
                                    input_idx_i_m = idx[m][k.replace('shortcut', 'conv1')][1]
                                    output_idx_i_m = idx_i[m]
                                elif 'linear' in k:
                                    input_idx_i_m = idx_i[m]
                                    output_idx_i_m = torch.arange(output_size, device=v.device)
                                else:
                                    raise ValueError('Not valid k')
                                idx[m][k] = (output_idx_i_m, input_idx_i_m)
                            else:
                                input_idx_i_m = idx_i[m]
                                idx[m][k] = input_idx_i_m
                        else:
                            input_size = v.size(0)
                            if 'linear' in k:
                                input_idx_i_m = torch.arange(input_size, device=v.device)
                                idx[m][k] = input_idx_i_m
                            else:
                                input_idx_i_m = idx_i[m]
                                idx[m][k] = input_idx_i_m
                    else:
                        pass
        elif cfg['model_name'] == 'transformer':
            idx_i = [None for _ in range(len(user_idx))]
            idx = [OrderedDict() for _ in range(len(user_idx))]
            for k, v in self.global_parameters.items():
                parameter_type = k.split('.')[-1]
                for m in range(len(user_idx)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if 'weight' in parameter_type:
                            if v.dim() > 1:
                                input_size = v.size(1)
                                output_size = v.size(0)
                                if 'embedding' in k.split('.')[-2]:
                                    output_idx_i_m = torch.arange(output_size, device=v.device)
                                    scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                    local_input_size = int(np.ceil(input_size * scaler_rate))
                                    input_idx_i_m = torch.arange(input_size, device=v.device)[:local_input_size]
                                    idx_i[m] = input_idx_i_m
                                elif 'decoder' in k and 'linear2' in k:
                                    input_idx_i_m = idx_i[m]
                                    output_idx_i_m = torch.arange(output_size, device=v.device)
                                elif 'linear_q' in k or 'linear_k' in k or 'linear_v' in k:
                                    input_idx_i_m = idx_i[m]
                                    scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                    local_output_size = int(np.ceil(output_size // cfg['transformer']['num_heads']
                                                                    * scaler_rate))
                                    output_idx_i_m = (torch.arange(output_size, device=v.device).reshape(
                                        cfg['transformer']['num_heads'], -1))[:, :local_output_size].reshape(-1)
                                    idx_i[m] = output_idx_i_m
                                else:
                                    input_idx_i_m = idx_i[m]
                                    scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                    local_output_size = int(np.ceil(output_size * scaler_rate))
                                    output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]
                                    idx_i[m] = output_idx_i_m
                                idx[m][k] = (output_idx_i_m, input_idx_i_m)
                            else:
                                input_idx_i_m = idx_i[m]
                                idx[m][k] = input_idx_i_m
                        else:
                            input_size = v.size(0)
                            if 'decoder' in k and 'linear2' in k:
                                input_idx_i_m = torch.arange(input_size, device=v.device)
                                idx[m][k] = input_idx_i_m
                            elif 'linear_q' in k or 'linear_k' in k or 'linear_v' in k:
                                input_idx_i_m = idx_i[m]
                                idx[m][k] = input_idx_i_m
                                if 'linear_v' not in k:
                                    idx_i[m] = idx[m][k.replace('bias', 'weight')][1]
                            else:
                                input_idx_i_m = idx_i[m]
                                idx[m][k] = input_idx_i_m
                    else:
                        pass
        else:
            raise ValueError('Not valid model name')
        return idx

    def distribute(self, user_idx):
        self.make_model_rate()
        param_idx = self.split_model(user_idx)
        local_parameters = [OrderedDict() for _ in range(len(user_idx))]
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(user_idx)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if 'weight' in parameter_type:
                        if v.dim() > 1:
                            local_parameters[m][k] = copy.deepcopy(v[torch.meshgrid(param_idx[m][k])])
                        else:
                            local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]]) # Extracting the sub-parameters for a layer. 
                    else:
                        local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])

                    if self.model_rate[user_idx[m]] == 0.25:
                        if k == 'layers.0.weight': # Need to do torch cat. 
                            updated_size = (int(np.ceil(v.size()[0] * self.model_rate[user_idx[m]])), v.size()[1])
                            self.initialize_weights(updated_size, k)
                        elif k == 'layers.2.weight': # Need to do torch cat along other dimension. 
                            updated_size = (v.size()[0], int(np.ceil(v.size()[1] * self.model_rate[user_idx[m]])))
                            self.initialize_weights(updated_size, k)
                        elif k == 'layers.0.bias': # Need to do torch cat. 
                            self.initialize_biases(int(np.ceil(v.size()[0] * self.model_rate[user_idx[m]])), k)
                            # local_parameters[m][k] = self.model_to_distribute[k]
                        elif k == 'layers.2.bias': # No need for torch cat. 
                            updated_size = v.size()[0]
                            self.initialize_biases(updated_size, k)
                        # In any case: 
                        local_parameters[m][k] = self.model_to_distribute[k]
                    else:
                        # Instead of the same initialization, vary the initialization of weights for epochs 1 and 2. 
                        # Use model_distributed + some small amount of uniform noise. 
                        if self.rd == 1:
                            local_parameters[m][k] = local_parameters[m][k].fill_(cfg['distribute_init_val']) 
                        else:
                            # self.rd == 2. 
                            mean_val = torch.mean(torch.abs(v))
                            # print("DISTRIBUTE: For epoch %s, mean_val = %s"%(self.rd, mean_val))
                            # print("DISTRIBUTE: For epoch %s, noise_std = %s and cfg['distribute_init_val'] = %s"%(self.rd, cfg['noise_scale'] * mean_val, cfg['distribute_init_val']))
                            if cfg['noise_scale'] is None:
                                local_parameters[m][k] = local_parameters[m][k].fill_(cfg['distribute_init_val'])
                            else:
                                local_parameters[m][k] = local_parameters[m][k].uniform_(cfg['distribute_init_val'] - cfg['noise_scale'] * mean_val, cfg['distribute_init_val'] + cfg['noise_scale'] * mean_val)
                    
                    # local_parameters[m][k] = local_parameters[m][k].fill_(0.25)
                    # local_parameters[m][k].uniform_(to=0.0010)
                    # torch.nn.init.xavier_normal_(local_parameters[m][k], gain=0.5)
                    # if (k in target_weights or k in target_biases):
                    # local_parameters[m][k].normal_(mean=1, std=0.0002)

                else:
                    local_parameters[m][k] = copy.deepcopy(v)

                
        # local_parameters: An array indexed by user_idx, which maps a key (e.g. blocks.weight.0 to its corresponding value (subset of global value))
        # param_idx: What is returned from split model (i.e. the indeces of which parameters to keep for a given layer). 
        return local_parameters, param_idx

    def generate_pts(self, num_elements, down_scale_factor=0.95, mu=0.0, sigma=0.5):
        vector = np.random.normal(mu, sigma, num_elements)
        abs_vector = abs(vector) * (-1)
        num_pos = np.floor(num_elements / 2).astype(int)
        pos_indices = np.random.choice(num_elements, num_pos, replace=False)
        negative_elements = np.delete(abs_vector, pos_indices)
        abs_vector[pos_indices] = -down_scale_factor * negative_elements  # set the negative values and turn them positive
        return abs_vector

    def initialize_weights(self, weight_shape, k):
        num_rows = weight_shape[0] # 250
        num_cols = weight_shape[1] # 784

        # print("INITIALIZE_PARAMS: num_rows = %s and num_cols = %s"%(num_rows, num_cols), flush=True)

        weights = np.zeros((num_rows, num_cols))
        for i in range(num_cols):
            weights[:, i] = self.generate_pts(num_rows, down_scale_factor=0.95, mu=0.0, sigma=0.5)
        # print("INITIALIZE_PARAMS: weights[:, 0] = %s"%(weights[:, 0]))
        
        self.model_to_distribute[k] = torch.from_numpy(weights)
    
    def initialize_biases(self, bias_shape, k):
        num_elems = bias_shape

        # print("INITIALIZE_BIASES: num_elems %s"%(num_elems), flush=True)

        biases = np.zeros(num_elems)
        biases = self.generate_pts(num_elems, down_scale_factor=0.95, mu=0.0, sigma=0.5)
        # print("INITIALIZE_PARAMS: biases head = %s"%(biases[:5]))
        
        self.model_to_distribute[k] = torch.from_numpy(biases)


    # Takes in local and global parameters for a particular key and outputs the gradient. 
    def compute_gradient(self, key, client, local_params, global_params):
        global_params_resized = copy.deepcopy(global_params)
        global_params_resized = global_params_resized[:local_params.size()[0]]
        user_grad = torch.subtract(global_params_resized, local_params)
        return user_grad

    def combine(self, local_parameters, param_idx, user_idx):
        count = OrderedDict()
        if cfg['model_name'] == 'fcnn':
            output_weight_name = [k for k in self.global_parameters.keys() if 'weight' in k][-1]
            output_bias_name = [k for k in self.global_parameters.keys() if 'bias' in k][-1]
            for k, v in self.global_parameters.items():
                parameter_type = k.split('.')[-1]
                count[k] = v.new_zeros(v.size(), dtype=torch.float32)
                tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
                my_tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
                for m in range(len(local_parameters)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if parameter_type == 'weight':
                            if v.dim() > 1:
                                if k == output_weight_name:
                                    label_split = self.label_split[user_idx[m]]
                                    param_idx[m][k] = list(param_idx[m][k])
                                    param_idx[m][k][0] = param_idx[m][k][0][label_split]
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                                else:
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                            else:
                                my_tmp_v += v
                                my_tmp_v[param_idx[m][k]] += user_grad
                                tmp_v[param_idx[m][k]] += local_parameters[m][k]
                                count[k][param_idx[m][k]] += 1
                        else:
                            if k == output_bias_name:
                                label_split = self.label_split[user_idx[m]]
                                param_idx[m][k] = param_idx[m][k][label_split]
                                tmp_v[param_idx[m][k]] += local_parameters[m][k][label_split]
                                count[k][param_idx[m][k]] += 1
                            else:
                                tmp_v[param_idx[m][k]] += local_parameters[m][k]
                                count[k][param_idx[m][k]] += 1
                    else:
                        tmp_v += local_parameters[m][k]
                        count[k] += 1
                tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
                v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
        elif cfg['model_name'] == 'conv':
            output_weight_name = [k for k in self.global_parameters.keys() if 'weight' in k][-1]
            output_bias_name = [k for k in self.global_parameters.keys() if 'bias' in k][-1]
            for k, v in self.global_parameters.items():
                parameter_type = k.split('.')[-1]
                count[k] = v.new_zeros(v.size(), dtype=torch.float32)
                tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
                my_tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
                for m in range(len(local_parameters)):
                    if (k == 'blocks.0.bias'):
                        user_grad = self.compute_gradient(k, user_idx[m], local_parameters[m][k], self.global_parameters[k])

                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if parameter_type == 'weight':
                            if v.dim() > 1:
                                if k == output_weight_name:
                                    label_split = self.label_split[user_idx[m]]
                                    param_idx[m][k] = list(param_idx[m][k])
                                    param_idx[m][k][0] = param_idx[m][k][0][label_split]
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                                else:
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                            else:
                                user_grad = self.compute_gradient(k, user_idx[m], local_parameters[m][k], self.global_parameters[k])

                                my_tmp_v += v
                                my_tmp_v[param_idx[m][k]] += user_grad
                                tmp_v[param_idx[m][k]] += local_parameters[m][k]
                                count[k][param_idx[m][k]] += 1

                                v_diff = torch.subtract(self.global_parameters[k], v)
                                grad_nongrad_diff = torch.subtract(my_tmp_v, tmp_v)
                        else:
                            if k == output_bias_name:
                                label_split = self.label_split[user_idx[m]]
                                param_idx[m][k] = param_idx[m][k][label_split]
                                tmp_v[param_idx[m][k]] += local_parameters[m][k][label_split]
                                count[k][param_idx[m][k]] += 1
                            else:
                                tmp_v[param_idx[m][k]] += local_parameters[m][k]
                                count[k][param_idx[m][k]] += 1
                    else:
                        tmp_v += local_parameters[m][k]
                        count[k] += 1
                tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
                v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
        elif 'resnet' in cfg['model_name']:
            for k, v in self.global_parameters.items():
                parameter_type = k.split('.')[-1]
                count[k] = v.new_zeros(v.size(), dtype=torch.float32)
                tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
                for m in range(len(local_parameters)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if parameter_type == 'weight':
                            if v.dim() > 1:
                                if 'linear' in k:
                                    label_split = self.label_split[user_idx[m]]
                                    param_idx[m][k] = list(param_idx[m][k])
                                    param_idx[m][k][0] = param_idx[m][k][0][label_split]
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                                else:
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                            else:
                                tmp_v[param_idx[m][k]] += local_parameters[m][k]
                                count[k][param_idx[m][k]] += 1
                        else:
                            if 'linear' in k:
                                label_split = self.label_split[user_idx[m]]
                                param_idx[m][k] = param_idx[m][k][label_split]
                                tmp_v[param_idx[m][k]] += local_parameters[m][k][label_split]
                                count[k][param_idx[m][k]] += 1
                            else:
                                tmp_v[param_idx[m][k]] += local_parameters[m][k]
                                count[k][param_idx[m][k]] += 1
                    else:
                        tmp_v += local_parameters[m][k]
                        count[k] += 1
                tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
                v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
        elif cfg['model_name'] == 'transformer':
            for k, v in self.global_parameters.items():
                parameter_type = k.split('.')[-1]
                count[k] = v.new_zeros(v.size(), dtype=torch.float32)
                tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
                for m in range(len(local_parameters)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if 'weight' in parameter_type:
                            if v.dim() > 1:
                                if k.split('.')[-2] == 'embedding':
                                    label_split = self.label_split[user_idx[m]]
                                    param_idx[m][k] = list(param_idx[m][k])
                                    param_idx[m][k][0] = param_idx[m][k][0][label_split]
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                                elif 'decoder' in k and 'linear2' in k:
                                    label_split = self.label_split[user_idx[m]]
                                    param_idx[m][k] = list(param_idx[m][k])
                                    param_idx[m][k][0] = param_idx[m][k][0][label_split]
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                                else:
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                            else:
                                tmp_v[param_idx[m][k]] += local_parameters[m][k]
                                count[k][param_idx[m][k]] += 1
                        else:
                            if 'decoder' in k and 'linear2' in k:
                                label_split = self.label_split[user_idx[m]]
                                param_idx[m][k] = param_idx[m][k][label_split]
                                tmp_v[param_idx[m][k]] += local_parameters[m][k][label_split]
                                count[k][param_idx[m][k]] += 1
                            else:
                                tmp_v[param_idx[m][k]] += local_parameters[m][k]
                                count[k][param_idx[m][k]] += 1
                    else:
                        tmp_v += local_parameters[m][k]
                        count[k] += 1
                tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
                v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
        else:
            raise ValueError('Not valid model name')

        return