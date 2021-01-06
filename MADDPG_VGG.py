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
device = torch.device( "cuda" if torch.cuda.is_available() else"cpu")  #"cuda" if torch.cuda.is_available() else
torch.cuda.set_device(0)
torch.set_default_tensor_type(torch.DoubleTensor)

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,padding=1)
        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3,padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv5 = nn.Conv2d(128,128, 3,padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3,padding=1)
        self.conv7 = nn.Conv2d(128, 128, 1,padding=1)
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.conv8 = nn.Conv2d(128, 256, 3,padding=1)
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv10 = nn.Conv2d(256, 256, 1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        self.conv11 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, 1, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()

        self.fc14 = nn.Linear(512*4*4,1024)
        self.drop1 = nn.Dropout2d()
        self.fc15 = nn.Linear(1024,1024)
        self.drop2 = nn.Dropout2d()
        self.fc16 = nn.Linear(1024,10)

    def forward(self, state):
        x = self.relu1(self.bn1(self.pool1(self.conv2(self.conv1(state)))))

        x = self.relu2(self.bn2(self.pool2(self.conv4(self.conv3(x)))))

        x = self.relu3(self.bn3(self.pool3(self.conv7(self.conv6(self.conv5(x))))))

        x = self.relu4(self.bn4(self.pool4(self.conv10(self.conv9(self.conv8(x))))))

        x = self.relu5(self.bn5(self.pool5(self.conv13(self.conv12(self.conv11(x))))))
        
        x = x.view(-1,512*4*4)
        output_1 = x
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        x = F.relu(self.fc15(x))
        x = self.drop2(x)
        output_2 = self.fc16(x).reshape(10)
        action = torch.max(output_2, 0)[1]
        dist = Categorical(output_2)
        action_logprob =dist.log_prob(action)
        #print("action: ",action,"action_logprob: ",action_logprob)
        
        return action.detach(),action_logprob,output_2.reshape(1,10),output_1.detach()  #action.detach()

class Critic(nn.Module):
    def __init__(self,agent_num):
        super(Critic, self).__init__()
        #cv
        self.cv_fc_1 = nn.Linear(512*4*4, 512)
        self.cv_fc_2 = nn.Linear(512, 100)
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

    agent_num = 3
    BATCH_SIZE = 1
    DOWNLOAD_CIFAR10 =True# False #
    original_CNN = False
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
                list_actions[i],output_2[i] = list_agents[i].select_action(b_x.double().to(device)) # float - int  [0,9]
            actions = torch.tensor(list_actions).double().to(device)
            reward = environment.reward_cal(actions,b_label,reward)
            rewards +=(sum(reward))
            #train
            for i in range(agent_num):
                a_loss = loss_func(output_2[i], torch.tensor(b_label).to(device))
                loss += list_agents[i].update(lr,reward[i],b_x.double().to(device),actions,a_loss)
            
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