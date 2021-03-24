"""utils.py

Author: Dany Vohl (2020-2021)
"""
import os
import pickle

def check_underscore(string):
    if string != '':
        if string[-1] != '_':
            string += '_'
    return string

def check_slash(string):
    if string != '':
        if string[-1] != '/':
            string += '/'
    return string

def save(variable, data, protocol=pickle.HIGHEST_PROTOCOL, state_prefix='', folder='states/'):
    if not os.path.exists(folder):
        os.makedirs(folder)

    if state_prefix != '':
        with open(check_slash(folder) + check_underscore(state_prefix) + variable + '.pickle', 'wb') as f:
            pickle.dump(data, f, protocol)
    else:
        with open(check_slash(folder) + variable + '.pickle', 'wb') as f:
            pickle.dump(data, f, protocol)

def load(variable, state_prefix='', folder='states/'):
    if state_prefix != '':
        if os.path.exists(folder + check_underscore(state_prefix) + variable + '.pickle'):
            with open(folder + check_underscore(state_prefix) + variable + '.pickle', 'rb') as f:
                return pickle.load(f)
        else:
            return None
    else:
        if os.path.exists(folder + variable + '.pickle'):
            with open(folder + variable + '.pickle', 'rb') as f:
                return pickle.load(f)
        else:
            return None
