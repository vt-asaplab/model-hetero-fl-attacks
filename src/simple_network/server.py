import copy
from collections import OrderedDict

import numpy as np
import ray
import torch
import constants
import model
import client

# Step 1: Distribute the model. 

class Server:
    # Server has a list of clients. 
    # Define the methods and datatypes. 
    def __init__(self, global_model, clients, client_capacities):
        self.num_clients = constants.NUM_CLIENTS
        self.global_model_history = []
        self.true_global_model = global_model
        self.advertised_global_model = model.Network()
        self.clients = clients
        self.client_capacities = client_capacities
        np.random.seed(42)
        torch.manual_seed(42)

    def fed_training(self):
        # Simply call fed_iteration for the number of training rounds. 
        for j in range(constants.NUM_ROUNDS):
            training_out = self.fed_iteration(j)
            
            for k, v in training_out.items():
                print("k = ", k)
                print("v = ", v)
            grad_wrt_weight = torch.subtract(training_out['linear1.weight'], self.true_global_model.state_dict()['linear1.weight'])
            print("grad_wrt_weight = %s"%(grad_wrt_weight))
            grad_wrt_bias = torch.subtract(training_out['linear1.bias'], self.true_global_model.state_dict()['linear1.bias'])
            print("grad_wrt_bias = %s"%(grad_wrt_bias))
            input_vector = torch.divide(grad_wrt_weight, grad_wrt_bias)
            print("input_vector = %s"%(input_vector))
            # print("gradient wrt weights = %s"%(training_out['self.linear1.weight']))
            # print("gradient wrt bias = %s"%(training_out['self.linear1.bias']))

            # Now that we have training out: Update the global state_dict to reflect this training. 
            self.true_global_model.load_state_dict(training_out)
            # print("self.true_global_model state_dict = ", self.true_global_model.state_dict())
            # print("self.true_global_model weight gradients = %s and bias gradients = %s"%(self.true_global_model.linear1.weight.grad, self.true_global_model.linear1.bias.grad))
            for name, param in self.true_global_model.named_parameters():
                print("name = %s"%(name))
                if (name == "linear1.weight"):
                    print("Grads for linear1 weight are %s"%(param.grad))
                elif (name == "linear1.bias"):
                    print("Grads for linear1 bias are %s"%(param.grad))


            self.global_model_history.append(training_out)
        
        # for j in range(constants.NUM_ROUNDS):
        #    print("global model history %s = %s"%(j, self.global_model_history[j]))

        # Now, extract a client's specific weight. 
        client_weights = self.extract_grad()
        print("Client 0's weights for linear 1 layer, to node 2 = ", client_weights)

            # OUT_PATH = f'state_dict_iter_{j}'
            # torch.save(self.true_global_model.state_dict(), OUT_PATH)
        
        # for j in range(constants.NUM_ROUNDS):
        #     IN_PATH = f'state_dict_iter_{j}'
        #     model.load_state_dict(torch.load(IN_PATH))

    def fed_iteration(self, j):
        # Consists of a single federated learning iteration. 
        # Includes distribution, local training, and aggregation phases. 
        
        # Step 1: Split the model. 
        j_hat = j % (constants.HIDDEN_LAYERS)
        client_models = self.split_model(j_hat)

        # Step 2: Call each client's local training method. 
        client_gradients = []
        dict_keys = set()
        for i in range(constants.NUM_CLIENTS):
            # print("---------------- For Client %s ----------------"%(i))
            # print("FED_ITERATION: client_models[i] = ", client_models[i])
            client_gradients.append(self.clients[i].train(constants.NUM_EPOCHS, client_models[i]))

        padded_gradients = []
        for i in range(constants.NUM_CLIENTS):
            # print(client_gradients[i])
            grad_copy = copy.deepcopy(client_gradients[i])

            for key in grad_copy.keys():
                dict_keys.add(key)
                if (key == 'linear1.weight'):
                    # print(grad_copy[key].shape)
                    # Create a new zero_tensor. 
                    # full_zero_tensor = torch.zeros((constants.HIDDEN_LAYERS, grad_copy[key].shape[1]))
                    new_size = (int(constants.HIDDEN_LAYERS - (self.client_capacities[i] * constants.HIDDEN_LAYERS)), grad_copy[key].shape[1])
                    # pad_start = (j_hat + (self.client_capacities[i] * constants.HIDDEN_LAYERS)) % constants.HIDDEN_LAYERS
                    # pre_padding = None
                    # post_padding = None
                    # mid_padding = None
                    # if (pad_start > j_hat):
                    #     post_padding = torch.zeros((constants.HIDDEN_LAYERS - pad_start), grad_copy[key].shape[1])
                    #     pre_padding = torch.zeros((j_hat, grad_copy[key].shape[1]))
                    # else:
                    #     mid_padding = torch.zeros((j_hat - pad_start), grad_copy[key].shape[1])
                        
                    # pre_padding = (int(constants.HIDDEN_LAYERS - (self.client_capacities[i] * constants.HIDDEN_LAYERS)), grad_copy[key].shape[1])

                    if (new_size[0] > 0):
                        zero_tensor = torch.zeros(new_size)
                        # pre_padding = full_zero_tensor[:j_hat, :]
                        # post_padding = full_zero_tensor[j_hat + int(self.client_capacities[i] * constants.HIDDEN_LAYERS):]
                        # print("zero_tensor = ", zero_tensor)
                        grad_copy[key] = torch.cat((grad_copy[key], zero_tensor))
                        grad_copy[key] = torch.roll(grad_copy[key], j_hat, 0)
                        # print("grad_copy[%s] for round %s after for client %s = %s"%(key, j_hat, i, grad_copy[key]))
                    # print("grad_copy size after lin1.weight = ", grad_copy[key].shape)
                elif (key == 'linear2.weight'):
                    new_size = (grad_copy[key].shape[0], int(constants.HIDDEN_LAYERS - (self.client_capacities[i] * constants.HIDDEN_LAYERS)))
                    if (new_size[1] > 0):
                        zero_tensor = torch.zeros(new_size)
                        grad_copy[key] = torch.cat((grad_copy[key], zero_tensor), dim=1)
                        grad_copy[key] = torch.roll(grad_copy[key], j_hat, 1)
                        # print("grad_copy[%s] for round %s after for client %s = %s"%(key, j_hat, i, grad_copy[key]))
                    # print("grad_copy size after lin2.weight = ", grad_copy[key].shape)
                elif (key == 'linear1.bias'):
                    new_size = int(constants.HIDDEN_LAYERS - (self.client_capacities[i] * constants.HIDDEN_LAYERS))
                    if (new_size > 0):
                        zero_tensor = torch.zeros(new_size)
                        # print("zero_tensor = ", zero_tensor)
                        grad_copy[key] = torch.cat((grad_copy[key], zero_tensor))
                        grad_copy[key] = torch.roll(grad_copy[key], j_hat)
                        # print("grad_copy[%s] for round %s after for client %s = %s"%(key, j_hat, i, grad_copy[key]))
                    # print("grad_copy size after lin1.bias = ", grad_copy[key].shape)
            
            padded_gradients.append(grad_copy)
            

        # Now, aggregate the corresponding weights and biases. 
        # self.aggregate(padded_gradients)
        aggregated_outputs = OrderedDict()
        for key in dict_keys:
            grad_values = []
            for i in range(self.num_clients):
                grad_values.append(padded_gradients[i][key])

            if (key == 'linear1.weight'):
                print("grad_values lin1 iteration %s before sending to aggregation = %s"%(j_hat, grad_values))

            # Aggregate all client tensors for a fixed key. 
            aggregated_outputs[key] = self.aggregate(grad_values)

        # for name in aggregated_outputs.keys():
        #     # print("AGGREGATION")
        #     print("aggregated_outputs name = ", name)
        #     print("aggregated_outputs val = ", aggregated_outputs[name])

        return aggregated_outputs
        # pass

    def split_model(self, j_hat):
        # Split the model according to each client's needs. 
        client_models = []
        for i in range(constants.NUM_CLIENTS):
            # Client i's data will be traindata[33i : 33(i+1)]
            # client_traindata = traindata[(constants.NUM_DATA_SAMPLES // constants.NUM_CLIENTS) * i : (constants.NUM_DATA_SAMPLES // constants.NUM_CLIENTS) * (i+1)]
            # client_model = model.Network_Scaled(self.client_capacities[i])
            client_model = self.true_global_model

            # print("Client ", i, "state dict = ", client_model.state_dict())

            # Now, copy over the appropriate entries in the global state_dict to each individual client's state dict. 
            # print("Global state_dict = ", clf.state_dict())
            global_state_dict = self.advertised_global_model.state_dict()
            if (j_hat == 1):
                self.advertised_global_model._init_weights_2(self.advertised_global_model.linear1)
                self.advertised_global_model._init_weights_2(self.advertised_global_model.linear2)
            print("SPLIT MODEL: For iteration %s, global model is %s"%(j_hat, global_state_dict))
            local_state_dict = copy.deepcopy(global_state_dict)

            for key in local_state_dict.keys():
                if (key == 'linear1.weight'):
                    if ((j_hat + self.client_capacities[i] * constants.HIDDEN_LAYERS) <= constants.HIDDEN_LAYERS):
                        local_state_dict[key] = global_state_dict[key][j_hat:j_hat+int(self.client_capacities[i]*constants.HIDDEN_LAYERS), :]
                        # print("IF CONDITION lin1: local_state_dict for client %s after slicing = %s"%(i, local_state_dict[key]))
                    else:
                        first_part = global_state_dict[key][:j_hat + int(self.client_capacities[i] * constants.HIDDEN_LAYERS) - constants.HIDDEN_LAYERS]
                        second_part = global_state_dict[key][j_hat:constants.HIDDEN_LAYERS]
                        local_state_dict[key] = torch.cat((first_part, second_part))
                        # print("ELSE CONDITION lin1: local_state_dict for client %s after slicing = %s"%(i, local_state_dict[key]))
                elif (key == 'linear2.weight'):
                    if ((j_hat + self.client_capacities[i] * constants.HIDDEN_LAYERS) <= constants.HIDDEN_LAYERS):
                        local_state_dict[key] = global_state_dict[key][:,j_hat:j_hat+int(self.client_capacities[i]*constants.HIDDEN_LAYERS) ]
                        # print("IF CONDITION lin2: local_state_dict for client %s after slicing = %s"%(i, local_state_dict[key]))
                    else:
                        first_part = global_state_dict[key][:, :j_hat + int(self.client_capacities[i] * constants.HIDDEN_LAYERS) - constants.HIDDEN_LAYERS]
                        second_part = global_state_dict[key][:, j_hat:constants.HIDDEN_LAYERS]
                        local_state_dict[key] = torch.cat((first_part, second_part), dim=1)
                        # print("ELSE CONDITION lin2: local_state_dict for client %s after slicing = %s"%(i, local_state_dict[key]))
                elif (key == 'linear1.bias'):
                    if ((j_hat + self.client_capacities[i] * constants.HIDDEN_LAYERS) <= constants.HIDDEN_LAYERS):
                        local_state_dict[key] = global_state_dict[key][j_hat:j_hat+int(self.client_capacities[i]*constants.HIDDEN_LAYERS)]
                        # print("IF CONDITION lin1 bias: local_state_dict for client %s after slicing = %s"%(i, local_state_dict[key]))
                    else:
                        first_part = global_state_dict[key][:j_hat + int(self.client_capacities[i] * constants.HIDDEN_LAYERS) - constants.HIDDEN_LAYERS]
                        second_part = global_state_dict[key][j_hat:constants.HIDDEN_LAYERS]
                        # print("ELSE CONDITION lin1 bias: local_state_dict for client %s after slicing = %s"%(i, local_state_dict[key]))
                else:
                    pass
                    # print(key)
                # print("lin2 bias: local_state_dict for client %s after slicing = %s"%(i, local_state_dict[key]))
            
            # Now, copy over state dicts for each client (use load_state_dict). 
            client_model.load_state_dict(local_state_dict)
            # print("Client %s state dict = %s"%(i, client_model.state_dict()))
            client_models.append(client_model)
            # print("client %s for iteration %s local state dict = %s"%(i, j_hat, client_model.state_dict()))

        return client_models
    
    def aggregate(self, client_params):
        # Inputs: NUM_CLIENTS tensors of the same dimensions. 
        # Output: Return the aggregated client tensors. 

        # Aggregate the global model based on the updates of clients. 
        # Server treats aggregation as a black box. 
        
        # First, create an ordered dict to store the results. 
        # aggregated_outputs = OrderedDict()

        params_stacked = torch.stack(client_params)
        # print("params_stacked = ", params_stacked)
        tensor_sum = torch.sum(params_stacked, dim=0)
        # print("tensor_sum = ", tensor_sum)
        # tensor_avg = torch.div(tensor_sum, constants.NUM_CLIENTS)
        # print("tensor_avg = ", tensor_avg)
        return tensor_sum
        # for i in range(len(client_params)):
        # print("client_params[%s] = %s"%(i, client_params[i]))

            # for key in client_params[i].keys():
            #    aggregated_outputs[key]
                
                # aggregated_outputs[]
                # Sum up the weights and biases. 


        # Simple aggregation scheme: Just add up the torch tensors and divide those tensors by the number of clients. 
        # pass

    def extract_grad(self):
        # Extract individual client gradients. 
        # In the first iteration, focus on linear 1 weights in the 3rd row. 
        lin1_iter1_node2 = self.global_model_history[0]['linear1.weight'][2:3, :]
        lin1_iter2_node2 = self.global_model_history[1]['linear1.weight'][2:3, :]
        # print("lin1_iter1 = ", lin1_iter1_node2, " and lin1_iter2 = ", lin1_iter2_node2)

        lin1_node2_diff = torch.sub(lin1_iter2_node2, lin1_iter1_node2)
        # print("lin1_node2_diff = ", lin1_node2_diff)

        return lin1_node2_diff
        # pass

