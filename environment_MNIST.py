import torch
import math
device = torch.device("cuda" if torch.cuda.is_available() else"cpu")  #"cuda" if torch.cuda.is_available() else 

class Environment_MNIST(object):
    def __init__(self,agent_num):
        self.agent_num = agent_num
        self.total = 0
        self.right = 0
        
    def reward_cal(self, actions,label,reward):
        reward=[ x*0 for x in reward ]
        self.total += 1
        vote = 0
        same = 0
        team_reward = 0.001 #0.01 too big
        
        # personal reward
        for i in range(self.agent_num):
            #reward[i]=(-math.log(abs((int(actions[i]))-label)+0.1) )/1000
            if bool(int(actions[i] )== label):
                vote += 1
                if self.agent_num==1:
                    self.right += 1 

        # team reward
        if self.agent_num>1:
            #  Are all results the same
            for i in range(self.agent_num):
                if actions[i] == actions[0]:
                    same = 1
                else:
                    same = 0
                    break
            if vote >= round(self.agent_num /2 ):
                reward =[ x+team_reward for x in reward ]
                if same == 1:
                    reward =[ x+team_reward for x in reward ]
                self.right += 1
            else:
                reward =[ x-team_reward for x in reward ]
                if same == 1:
                    reward =[ x - team_reward for x in reward ]
        return reward

    def accuracy_rate(self):
        rate = float(self.right/self.total )
        return rate

    def reset(self):
        self.total = 0
        self.right = 0
    
    def test(self,list_agents,test_x,test_y):
        right = 0
        for x,y in zip(test_x,test_y):
            vote = 0
            for i in range(self.agent_num):
                pred_y,_ = list_agents[i].select_action(x.reshape(1,1,28,28).double().to(device))
                if int(pred_y) == int(y):
                    vote += 1
                    if self.agent_num ==1:
                        right += 1
           
            if vote >= round(self.agent_num /2 ) and self.agent_num>1:
                right += 1

        return float(right/2000)