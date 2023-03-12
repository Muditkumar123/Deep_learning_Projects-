# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random #we will be taking random samples from the different batches while implementing experience replay
import os # model load and save ke liye
import torch #neural network ke liye
import torch.nn as nn
import torch.nn.functional as F #contain different function while implementing NN
import torch.optim as optim  #optimizer to  perform stochastic gradient descent
import torch.autograd as autograd #we are taking variable class from autograd we are imporing variable class to make some conversion , tensor with gradient
from torch.autograd import Variable

# Creating the architecture of the Neural Network

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action): #input neurons (5-dim) with 3 signals +,- orientation,output : 3 actions
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30) #first layer connecting input layer to hidden one ,30 in hidden layer
        self.fc2 = nn.Linear(30, nb_action)#hidden layer to output one
    
    def forward(self, state):#state are input entering neural network
        Y = F.relu(self.fc1(state)) # Y represent hidden neurons which  will active using rectifier function(relu) ,fc1 ke under input state dali h
        q_values = self.fc2(Y)  # output layer , q values h ye and softmax use kiya h as default
        return q_values

# Implementing Experience Replay

class ReplayMemory(object):
    
    def __init__(self, capacity): # previous capacity lets say c= 100 so pichle 100 states rkhega ye memory me experience replay ke liye
        self.capacity = capacity
        self.memory = [] # yha store hoge transitions
    
    def push(self, event): # event or transition h,event=(st(last state),st+1(new state),at(last action that displayed),last reward obtain)
        self.memory.append(event)
        if len(self.memory) > self.capacity:# deleting the oldest transition from memory
            del self.memory[0]
    
    def sample(self, batch_size):#random samples lege memory se
        samples = zip(*random.sample(self.memory, batch_size))#random from library se sample function liya h,jo random sample lega memory se jiska fixed size = batch_soze h,zip(*list)reshape list eg:l=[(1,2,3),(4,5,6)] ,zip(*l)=[(1,4),(2,3),(5,6)] why used this? coz ye list (state,action and reward h) so zip krke [(action1,action2),(reward1,reward2) , ye batches pytorch variable me jaege jo tensor and gradient leta h
        return map(lambda x: Variable(torch.cat(x, 0)), samples)#concatinate the samples to 1-D, batches will be well alligned wrt time

# Implementing Deep Q Learning

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):#gamma is delay cofficient
        self.gamma = gamma
        self.reward_window = []# here mean of rewards of 100 actions
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)# lr=learning rate
        self.last_state = torch.Tensor(input_size).unsqueeze(0)#network only accept batch of input observations,pytorch wants a tensor vector but with extra dimension that corresponds to the batch,extra dim corresponding to the batch will be first dim of last variable
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):# right action decide krna h car ko whether go left,right or straight, it depends upon output layer Q values which in short depends upon input state
        probs = F.softmax(self.model(Variable(state, volatile = True))*75) # T=100 #converting torch sensor into torch variable without gradient decent,softmax Q values ko probablity distribution me convert krdega 0-1 ke bich me,saving memory,T permameter helps in deciding the action of agent ,more the T is higher  ,more sure the NN will take that action 
        action = probs.multinomial(num_samples=1)#gives random draw from prob  ,also this contain that tensor extra dimension
        return action.data[0,0]# this contain data ,(stackoverflow)
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action): #batch_state is current state,this is transition of markov decision process
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)# why gather used as  without it 0,1,2 tino possible action dega but we are interested in only 1 action decided by the model , batch state and action have different dim to using unsqueeze on action make them equal(source:stackoverflow),using squeeze we are killing fake dimension as simple output chaiye
        next_outputs = self.model(batch_next_state).detach().max(1)[0]#from algorithm,this line of code is from stackoverflow, detaching all the outputs of the model?,Q values is rt to  action so max(1),for q value of st+1 ,0->state

        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)#this is huber loss function
        self.optimizer.zero_grad()# re initialize the optimizer at each iteration of the loop
        td_loss.backward(retain_graph = True)# memory free krne ke liye, this is backpropagation
        self.optimizer.step()# this update the weights
    
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)#signal is the state,updating new state,new state depends upon the signal detected,signal is list to 5 elements to in order to add it in neural network convert it in torch.tensor
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward]))) #adding transition to the memory, these variables are taken from the deep-Q-algo
        action = self.select_action(new_state)#after reaching the new state,action lena h agent ko and ye input ka output dega,shi decision lene me
        if len(self.memory.memory) > 100:#first memory  is object of replaymemory class,2nd one is at tribute of self,memory
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)#gonna return  the bacthes respectively ,100 transitions se liye gye h ye
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action#isko bhi update krna h
        self.last_state = new_state
        self.last_reward = reward#inko update krna pdega ,warna error aya tha
        self.reward_window.append(reward)#going to keep  track of how traing is going by taking mean of last 100 rewards
        if len(self.reward_window) > 1000:#fixing the size
            del self.reward_window[0]
        return action# this suppose the return the action that was displayed when reaching the new state
    
    def score(self):# this function is to compute/mean of all  rewards in reward window
        return sum(self.reward_window)/(len(self.reward_window)+1.) # +1 isliye taki denominator kbhi 0 na h
    
    def save(self):#saving the model(nn,optimizer),we wanna save the last weights updated at the last iteration,optimizer as it is connected to the weights 
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'brain.pth')
    
    def load(self):#loading the model
        if os.path.isfile('last_brain.pth'):
            print("loading Savepoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])#updating the weight and parameter of model 
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Done...")
        else:
            print("No Savepoint found...")
