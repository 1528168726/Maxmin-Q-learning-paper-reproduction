import os
import sys
import argparse

from utils.helper import make_dir
from experiment import Experiment


def get_num_combinations_of_dict(config_dict):
    assert type(config_dict) == dict, 'Config file must be a dict!'
    num_combinations_of_dict = 1
    for key, values in config_dict.items():
        num_combinations_of_list = get_num_combinations_of_list(values)
        num_combinations_of_dict *= num_combinations_of_list
    config_dict['num_combinations'] = num_combinations_of_dict


def get_num_combinations_of_list(config_list):
    assert type(config_list) == list, 'Elements in a config dict must be a list!'
    num_combinations_of_list = 0
    for value in config_list:
        if type(value) == dict:
            if not ('num_combinations' in value.keys()):
                get_num_combinations_of_dict(value)
            num_combinations_of_list += value['num_combinations']
        else:
            num_combinations_of_list += 1
    return num_combinations_of_list


def generate_config_for_idx(config_dicts, idx):
    # Get config dict given the index
    cfg = get_dict_value(config_dicts, (idx - 1) % config_dicts['num_combinations'])
    # Set config index
    cfg['config_idx'] = idx
    # Set number of combinations
    cfg['num_combinations'] = config_dicts['num_combinations']
    return cfg


def get_list_value(config_list, idx):
    for value in config_list:
        if type(value) == dict:
            if idx + 1 - value['num_combinations'] <= 0:
                return get_dict_value(value, idx)
            else:
                idx -= value['num_combinations']
        else:
            if idx == 0:
                return value
            else:
                idx -= 1

def get_dict_value(config_dict, idx):
    cfg = dict()
    for key, values in config_dict.items():
        if key == 'num_combinations':
            continue
        num_combinations_of_list = get_num_combinations_of_list(values)
        value = get_list_value(values, idx % num_combinations_of_list)
        cfg[key] = value
        idx = idx // num_combinations_of_list
    return cfg

def main(argv):
  #config of catcher
  config_dicts = {'env':[{'name': ["Catcher-PLE-v0"],'max_episode_steps': [2000],'input_type': ["feature"]}],
         'agent':[{'name':["QLearning","DQLearning"]},{"name": ["MaxminDQN"],"target_networks_num": [2,3,4,5,6,7,8,9]}],
         'train_steps': [3.5e6],'target_network_update_frequency': [200],
         'test_per_episodes': [-1], 'network_update_frequency':1,'display_interval': [100],
         'generate_random_seed': [True],'seed': [1],'render':False,'discount': [0.99],'device': ["cpu"],
         'epsilon_steps': [1e3],'epsilon_start': [1.0],'epsilon_end': [0.01],'epsilon_decay': [0.999]}

  get_num_combinations_of_dict(config_dicts)
  # Set experiment name and log paths
  cfg = generate_config_for_idx(config_dicts,1)
  cfg['env'].setdefault('max_episode_steps', -1)
  cfg.setdefault('show_tb', False)
  cfg.setdefault('render', False)
  cfg.setdefault('gradient_clip', -1)
  cfg.setdefault('hidden_act', 'ReLU')
  cfg.setdefault('output_act', 'Linear')

  make_dir(f"./logs/{cfg['exp']}/{cfg['config_idx']}/")
  cfg['train_log_path'] = cfg['logs_dir'] + 'result_Train.feather'
  cfg['test_log_path'] = cfg['logs_dir'] + 'result_Test.feather'
  cfg['model_path'] = cfg['logs_dir'] + 'model.pt'
  cfg['cfg_path'] = cfg['logs_dir'] + 'config.json'

  exp = Experiment(cfg)
  exp.run()

if __name__=='__main__':
  main(sys.argv)