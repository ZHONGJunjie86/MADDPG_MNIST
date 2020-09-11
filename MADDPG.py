from utils import cross_loss_curve, initialization
from environment_MNIST import Environment_MNIST
import torch.utils.data as Data
import torchvision
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import warnings
import random

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else"cpu")  #"cuda" if torch.cuda.is_available() else
torch.set_default_tensor_type(torch.DoubleTensor)

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d( in_channels=1,  out_channels=16, kernel_size=5, stride=1,padding=2,),   # output shape (16, 28, 28)
            nn.BatchNorm2d(16),
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2, stride=2),    #  output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.BatchNorm2d(32),
            nn.ReLU(),                 
            nn.MaxPool2d(2, stride=2),                # output shape (32, 7, 7)
        )
        self.output_1 = nn.Linear(32 * 7 * 7, 1000) 
        self.output_2 = nn.Linear(1000,10)   # output 10 classes

    def forward(self, state):
        x = self.conv1( state)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           #  (batch_size, 32 * 7 * 7)
        output_1 = self.output_1(x)
        output_2 = self.output_2(output_1).reshape(10)
        action = torch.max(output_2, 0)[1]
        dist = Categorical(output_2)
        action_logprob =dist.log_prob(action)
        #print("action: ",action,"action_logprob: ",action_logprob)
        
        return action.detach(),action_logprob,output_2.reshape(1,10)  

class Critic(nn.Module):
    def __init__(self,agent_num):
        super(Critic, self).__init__()
        #cv
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d( in_channels=1,  out_channels=16, kernel_size=5, stride=1,padding=2,),   # output(16, 28, 28)
            nn.BatchNorm2d(16),
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2, stride=2),    #  output(16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input(16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output(32, 14, 14)
            nn.BatchNorm2d(32),
            nn.ReLU(),                 
            nn.MaxPool2d(2, stride=2),        # output (32, 7, 7)
        )
        self.cvlinear_1 = nn.Linear(32 * 7 * 7, 1000) 
        self.cvlinear_2 = nn.Linear(1000,100)  
        #num
        self.agent_num = agent_num
        self.linear1 = nn.Linear(self.agent_num, 128)
        self.linear2 = nn.Linear(128, 100)
        #
        self.linear3 = nn.Linear(200, 64)
        self.linear4 = nn.Linear(64, 1)

    def forward(self,state,actions):
        # cv
        x = self.conv1( state)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x  = self.cvlinear_1(x)
        x = self.cvlinear_2(x)
        # num
        output_1 = F.relu(self.linear1(actions.reshape(1,self.agent_num))) 
        output_2 = F.relu(self.linear2(output_1)).reshape(1,100)
        # combine
        output_2 = torch.cat((x,output_2),1) 
        output_3 = F.relu(self.linear3(output_2))
        value = torch.tanh(self.linear4(output_3))

        return value

class DDPG:
    def __init__(self, lr,agent_num,original_CNN):
        self.lr = lr
        self.action_logprob = 0
        self.original_CNN = original_CNN
        
        self.actor = Actor().to(device)
        self.critic = Critic(agent_num).to(device)
        self.A_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, betas=(0.95, 0.999)) 
        self.C_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=(0.95, 0.999)) 

    def select_action(self, state):
        action,self.action_logprob,output_2 = self.actor(state)
        action = torch.clamp(action.detach(), 0, 9)  #[0,9]
        return action,output_2 #.cpu().data.numpy().flatten()  
    
    def update(self, lr,reward,state,actions,a_loss_MSELoss):
                
        with torch.autograd.set_detect_anomaly(True):
            if self.original_CNN == True:
                self.A_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, betas=(0.95, 0.999)) 
                a_loss =a_loss_MSELoss
                self.A_optimizer.zero_grad()
                a_loss.backward() 
                self.A_optimizer.step()
                return torch.tensor(0)
            else:
                reward = torch.tensor(reward).to(device)
                self.A_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, betas=(0.95, 0.999)) 
                self.C_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=(0.95, 0.999)) 
                # Evaluating old actions and values :
                state_values= self.critic(state,actions).squeeze()
                advantage = reward.detach()  - state_values  
                #critic 
                c_loss = (reward.detach()  - state_values).pow(2) 
                self.C_optimizer.zero_grad()
                c_loss.backward()
                self.C_optimizer.step()
                #actor
                a_loss =  a_loss_MSELoss 
                self.A_optimizer.zero_grad()
                a_loss.backward(retain_graph=True) 
                self.A_optimizer.step()
                a_loss =-(self.action_logprob * advantage.detach())  
                self.A_optimizer.zero_grad()
                a_loss.backward() 
                self.A_optimizer.step()
                return advantage.pow(2) 

def main():
    ################### initialization ########################
    epoch = 0
    step_real = 0

    sample_lr = [
        0.0001, 0.00009, 0.00008, 0.00007, 0.00006, 0.00005, 0.00004, 0.00003,
        0.00002, 0.00001, 0.000009, 0.000008, 0.000007, 0.000006, 0.000005,
        0.000004, 0.000003, 0.000002, 0.000001
    ]
    lr = 0.0001
    if step_real >=400 : 
        try:
            lr = sample_lr[int(step_real//400)]*10
        except(IndexError):
            lr = 0.000001* (0.9 ** ((step_real -3000) // 3000))
    
    DOWNLOAD_MNIST = False     # choose download dataset
    original_CNN = True #False #      #choose to  use MAS or original CNN
    
    agent_num = 5     # the number of agents
    BATCH_SIZE = 1
    list_agents = []
    list_actions = []
    max_index = []
    reward = []
    output_2 = []
    loss_func = nn.CrossEntropyLoss()    
    for i in range(agent_num):
        list_actions.append(0)
        reward.append(0)
        max_index.append(0)
        output_2.append(0)
        list_agents.append(DDPG(lr,agent_num,original_CNN))

    weight_path = os.path.abspath(os.curdir)+'/Tasks/A8_MAS_MNIST/weight/'
    save_curve_pic = os.path.abspath(os.curdir)+'/Tasks/A8_MAS_MNIST/result/loss_curve.png'
    save_critic_loss = os.path.abspath(os.curdir)+'/Tasks/A8_MAS_MNIST/training_data/loss.csv'
    save_reward = os.path.abspath(os.curdir)+'/Tasks/A8_MAS_MNIST/training_data/reward.csv'
    save_accuracy = os.path.abspath(os.curdir)+'/Tasks/A8_MAS_MNIST/training_data/accuracy.csv'
    initialization(list_agents,weight_path,agent_num)

    rewards =0
    loss = 0
    environment = Environment_MNIST(agent_num)
    train_data = torchvision.datasets.MNIST(root=os.path.abspath(os.curdir)+"/Tasks/mnist",train=True,                                    
        transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST,)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_data = torchvision.datasets.MNIST(root=os.path.abspath(os.curdir)+"/Tasks/mnist", train=False)
    test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.DoubleTensor)[:2000]/255.
    test_y = test_data.test_labels[:2000]
    ##################  start  #########################
    while epoch<1:
        for step, (b_x, b_label) in enumerate(train_loader):
            #choose action
            for i in range(agent_num):
                list_actions[i],output_2[i] = list_agents[i].select_action(b_x.double().to(device))
            actions = torch.tensor(list_actions).double().to(device)
            reward = environment.reward_cal(actions,b_label,reward)
            rewards +=(sum(reward))
            #train
            for i in range(agent_num):
                a_loss = loss_func(output_2[i], torch.tensor(b_label).to(device))
                loss += list_agents[i].update(lr,reward[i],b_x.double().to(device),actions,a_loss)
        
             # draw loss_curve
            if step_real % 200== 0 and step_real!=0:
                accuracy = environment.test(list_agents,test_x,test_y)
                print('step: ', step_real, "accuracy_rate:",accuracy,'| train loss: %.4f' % loss," LR: ",lr)
                cross_loss_curve(loss.detach().cpu()/agent_num,rewards/(agent_num),accuracy*10,save_curve_pic,save_critic_loss,save_reward,save_accuracy )  #
                for i in range(agent_num):
                    torch.save(list_agents[i].actor.state_dict(),weight_path +"actor"+str(i)+".pkl")
                    torch.save(list_agents[i].critic.state_dict(),weight_path +"critic"+str(i)+".pkl")
                if step_real >=400 : 
                    try:
                        lr = sample_lr[int(step_real//400)]*10
                    except(IndexError):
                        lr = 0.000001* (0.9 ** ((step_real -3000) // 3000))
                rewards = 0
                loss = 0
                environment.reset()
            step_real += 1

        epoch += 1

    return None 

if __name__ == '__main__':
    main()