# Model-Heterogeneous-FL-Attacks
## Requirements: 
Refer to requirements.txt. To install necessary packages, run ``pip install -r requirements.txt``

## Code Structure: 
- train_classifier_fed.py: High level source file for the Convergence Rate Attack. 
- train_classifier_rolex.py: High level source file for the Rolling Model Attack. 
- fed.py: Controls submodel distribution/aggregation for the Convergence Rate Attack. 
- fed_rolex.py: Controls submodel distribution/aggregation for the Rolling Model Attack. 

## Parameters: 
- Hyperparameters are set through the config.yml and utils.py files. 
- To specify an output directory for the Tables, change cfg['file_output'] for the attack you are running (defined in train_classifier_fed.py and train_classifier_rolex.py). 
- To alter the local training dataset size, change cfg['local_train_size']
- For Rolling Model Attack: 
    - Change the noise percentage with cfg['noise_scale']
    - Change the initial distribution value with cfg['distribute_init_val']

## How to Run: 
Navigate to source directory. Then, run the desired attack using one of the commands below.  

Convergence Rate Attack: python3 train_classifier_fed.py --data_name MNIST --model_name fcnn --control_name 1_5_1_non-iid-2_fix_a1-b1-c1_bn_1_1

Rolling Model Attack: python3 train_classifier_rolex.py --data_name MNIST --model_name fcnn --control_name 1_3_1_iid_fix_a1-b1-c1_bn_1_1

## Output: 
- The table file will summarize the relevant attack metrics.