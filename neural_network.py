# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 15:10:21 2018

@author: MarcoBz

"""

import numpy as np
import scipy.special 

class Neural_network:
    
    def __init__(self, input_nodes, output_nodes, hidden_nodes, hidden_layers, learning_rate, function):
        
        self.input_n = input_nodes
        self.output_n = output_nodes
        self.hidden_n = hidden_nodes
        self.hidden_l = hidden_layers
        self.total_layers = self.hidden_l + 2 
        self.weights_matrixes = [None] * (self.total_layers - 1)
        self.layers_inputs = [None] * self.total_layers
        self.layers_outputs = [None] * self.total_layers
        self.lr = learning_rate
        self.function = function
        self.create_all_weights_matrixes()
        if self.function == "sigmoid":
            self.activation_function = lambda x: scipy.special.expit(x)
        
    
    def train(self, inputs, targets):
        
        check_dim = True
        
        if len(inputs) != len(targets):
            print("different number of input and target sets")
            check_dim = False
        
        for i in range(len(inputs) - 1):
            if len(inputs[i]) != len(inputs[i + 1]):
                print("error in input data of trainset")
                check_dim = False
                break
        
        if check_dim and len(inputs[0]) != self.input_n:
            print("error in input data of trainset")
            check_dim = False
            
        for i in range(len(targets) - 1):
            if len(targets[i]) != len(targets[i + 1]):
                print("error in target data of trainset")
                check_dim = False
                break
        
        if check_dim and len(targets[0]) != self.output_n:
            print("error in target data of trainset")
            check_dim = False
            
        if check_dim:
            self.trainset_dim = len(inputs)
            self.trainset_inputs = [None] * self.trainset_dim
            self.trainset_targets = [None] * self.trainset_dim
            self.trainset_outputs = [None] * self.trainset_dim
            self.layers_train_inputs = []
            self.layers_train_outputs = []
            self.errors_train = []
            for i in range(self.trainset_dim):
                self.trainset_inputs[i] = np.array(inputs[i], ndmin = 2).T
                self.trainset_targets[i] = np.array(targets[i], ndmin = 2).T
                
                for j in range(self.total_layers):
                    self.get_layer_train(i,j)
                self.trainset_outputs[i] = self.layers_train_outputs[i][-1]
                
                for j in range(self.total_layers - 2, -1, -1):
                    self.get_errors(i,j)
                    
                for j in range(self.total_layers - 2, -1, -1):
                    self.update_weights(i,j)
        else:
            print("no training")
            
    
    def query(self, inputs):
        
        self.inputs = np.array(inputs, ndmin = 2).T
        for i in range(self.total_layers):
            self.get_layer(i)
        self.outputs = self.layers_outputs[-1]
        
        return self.outputs
    
    def create_all_weights_matrixes(self):
        
        for n in range(self.total_layers - 1):
            self.create_weights_matrix(n)
    
    def create_weights_matrix(self, layer):
        #array, first is input_l - first_hidden_l, last is last_hidden_l - output_l
        if layer == 0:
            #input_l - first_hidden_l
            current_matrix = np.random.normal(0.0, pow(self.hidden_n, -0.5), (self.hidden_n, self.input_n)) 
            #current_matrix = np.ones((self.hidden_n, self.input_n))
            
        elif layer == self.total_layers - 2:
            #last_hidden_l - output_l
            current_matrix = np.random.normal(0.0, pow(self.output_n, -0.5), (self.output_n, self.hidden_n)) 
            #current_matrix = np.ones((self.output_n, self.hidden_n))
            
        else:
            #any_hidden_l - any_hidden_l
            current_matrix = np.random.normal(0.0, pow(self.hidden_n, -0.5), (self.hidden_n, self.hidden_n)) 
            #current_matrix = np.ones((self.hidden_n, self.hidden_n))
        
        self.weights_matrixes[layer] = current_matrix
        
    def get_layer(self, i):
        
        if i == 0:
            self.layers_inputs[0] = self.inputs
            self.layers_outputs[0] = self.inputs
        else:
            self.layers_inputs[i] = np.dot(self.weights_matrixes[i - 1], self.layers_outputs[i - 1])
            self.layers_outputs[i] = self.activation_function(self.layers_inputs[i])
         
    def get_layer_train(self, i, j):
        
        if j == 0:
            self.layers_train_inputs.append([None] * self.total_layers)
            self.layers_train_outputs.append([None] * self.total_layers)
            self.layers_train_inputs[i][0] = self.trainset_inputs[i]
            self.layers_train_outputs[i][0] = self.trainset_inputs[i]
        else:
            self.layers_train_inputs[i][j] = np.dot(self.weights_matrixes[j - 1], self.layers_train_outputs[i][j - 1])
            self.layers_train_outputs[i][j] = self.activation_function(self.layers_train_inputs[i][j])
    
    def get_errors(self, i, j):
        
        if j == self.total_layers - 2:
            self.errors_train.append([None] * (self.total_layers - 1))
            self.errors_train[i][-1] = self.trainset_targets[i] - self.trainset_outputs[i]
        else:
            self.errors_train[i][j] = np.dot(self.weights_matrixes[j + 1].T, self.errors_train[i][j + 1])
            
    def update_weights(self, i, j):
        
        self.weights_matrixes[j] += self.lr * np.dot((self.errors_train[i][j] * self.layers_train_outputs[i][j + 1] * (1 - self.layers_train_outputs[i][j + 1])), np.transpose(self.layers_train_outputs[i][j]))
            
            
            
            