from utils import cross_loss_curve, initialization
from environment_CIFAR import Environment_CIFAR
import torch.utils.data as Data
import torchvision
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.distributions import Categorical
import numpy as np
import warnings

warnings.filterwarnings("ignore")
device = torch.device( "cpu")  #"cuda" if torch.cuda.is_available() else
torch.cuda.set_device(0)
torch.set_default_tensor_type(torch.DoubleTensor)

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
 
    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Actor(nn.Module):
    def __init__(self,ResidualBlock = ResidualBlock):
        super(Actor, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, 10)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, state):
        out = self.conv1(state)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        output_2 = self.fc(out).reshape(10)
        action = torch.max(output_2, 0)[1]
        dist = Categorical(output_2)
        action_logprob =dist.log_prob(action)
        #print("action: ",action,"action_logprob: ",action_logprob)
        
        return action.detach(),action_logprob,output_2.reshape(1,10) ,out.detach() #action.detach()

class Critic(nn.Module):
    def __init__(self,agent_num):
        super(Critic, self).__init__()
        #cv
        self.cv_fc_1 = nn.Linear(512, 100)
        self.cv_fc_2 = nn.Linear(100, 100)
        #num
        self.agent_num = agent_num
        self.linear1 = nn.Linear(self.agent_num, 128)
        self.linear2 = nn.Linear(128, 100)
        #
        self.linear3 = nn.Linear(200, 64)
        self.linear4 = nn.Linear(64, 1)

    def forward(self,state,actions):
        # cv
        x=  F.relu(self.cv_fc_1(state))
        x=  F.relu(self.cv_fc_2(x))
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
        self.original_CNN = False
        
        self.actor = Actor().to(device)
        self.critic = Critic(agent_num).to(device)
        self.A_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, betas=(0.95, 0.999)) 
        self.C_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=(0.95, 0.999)) 

    def select_action(self, state):
        action,self.action_logprob,output_2 ,output_1= self.actor(state)
        action = torch.clamp(action.detach(), 0, 9)  #[0,9]
        return action,output_2,output_1#.cpu().data.numpy().flatten()  
    
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
                c_loss = (reward.detach()  - state_values).pow(2) 
                self.C_optimizer.zero_grad()
                c_loss.backward()
                self.C_optimizer.step()
                #
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
    lr = 0.01
    if step_real >=1000 :
        try:
            lr = sample_lr[int(step_real//1000)]*100
        except(IndexError):
            try:
                lr = sample_lr[int(step_real//4000)]#0.0001* (0.9 ** ((step_real -1500) //1500))
            except(IndexError):
                0.000001*(0.9 ** ((step_real -10000) //10000))

    agent_num = 5
    BATCH_SIZE = 1
    DOWNLOAD_CIFAR10 =False #True# 
    original_CNN = False
    list_agents = []
    list_actions = []
    max_index = []
    reward = []
    share_A_C = []
    output_2 = []
    loss_func = nn.CrossEntropyLoss()    
    for i in range(agent_num):
        list_actions.append(0)
        reward.append(0)
        max_index.append(0)
        output_2.append(0)
        list_agents.append(DDPG(lr,agent_num,original_CNN))
        share_A_C.append(0)

    weight_path = os.path.abspath(os.curdir)+'/Tasks/A7_MAS_CIFAR10/weight/'
    save_curve_pic = os.path.abspath(os.curdir)+'/Tasks/A7_MAS_CIFAR10/result/loss_curve.png'
    save_critic_loss = os.path.abspath(os.curdir)+'/Tasks/A7_MAS_CIFAR10/training_data/loss.csv'
    save_reward = os.path.abspath(os.curdir)+'/Tasks/A7_MAS_CIFAR10/training_data/reward.csv'
    save_accuracy = os.path.abspath(os.curdir)+'/Tasks/A7_MAS_CIFAR10/training_data/accuracy.csv'
    initialization(list_agents,weight_path,agent_num)

    rewards =0
    loss = 0
    environment = Environment_CIFAR(agent_num)
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    train_data = torchvision.datasets.CIFAR10(root=os.path.abspath(os.curdir)+"/Tasks/CIFAR10",train=True,                                    
                                               transform=transform, download=DOWNLOAD_CIFAR10,)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_data =  torchvision.datasets.CIFAR10(root=os.path.abspath(os.curdir)+"/Tasks/CIFAR10",train = False,
                                        download=DOWNLOAD_CIFAR10, transform=transform)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)
    ##################  start  #########################
    while epoch<10:
        for step, (b_x, b_label) in enumerate(train_loader):
            #choose action
            for i in range(agent_num):
                list_actions[i],output_2[i] ,share_A_C[i]= list_agents[i].select_action(b_x.double().to(device)) # float - int  [0,9]
            actions = torch.tensor(list_actions).double().to(device)
            reward = environment.reward_cal(actions,b_label,reward)
            rewards +=(sum(reward))
            #train
            for i in range(agent_num):
                a_loss = loss_func(output_2[i], torch.tensor(b_label).to(device))
                loss += list_agents[i].update(lr,reward[i],share_A_C[i],actions,a_loss) #b_x.double().to(device)
            
             # draw loss_curve
            if step_real % 200 == 0 and step_real !=0 :
                accuracy = environment.test(list_agents,test_loader)#.accuracy_rate()
                print('step: ', step, "accuracy_rate:",accuracy,'| train loss: %.4f' % loss," LR: ",lr)
                cross_loss_curve(loss.detach().cpu()/agent_num,rewards/(agent_num*20),accuracy*10,save_curve_pic,save_critic_loss,save_reward,save_accuracy )  #
                for i in range(agent_num):
                    torch.save(list_agents[i].actor.state_dict(),weight_path +"actor"+str(i)+".pkl")
                    torch.save(list_agents[i].critic.state_dict(),weight_path +"critic"+str(i)+".pkl")
                if step_real >=1000 :
                    try:
                        lr = sample_lr[int(step_real//1000)]*100
                    except(IndexError):
                        try:
                            lr = sample_lr[int(step_real//4000)]#0.0001* (0.9 ** ((step_real -1500) //1500))
                        except(IndexError):
                            0.000001*(0.9 ** ((step_real -10000) //10000))
                rewards = 0
                loss = 0
                environment.reset()
            step_real += 1

        epoch += 1

    return None 

if __name__ == '__main__':
    main()