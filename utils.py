import matplotlib.pyplot as plt
import numpy as np
import time,random
import torch
import os 

def cross_loss_curve(critic_loss,total_rewards,accuracy,save_curve_pic,save_critic_loss,save_reward,save_accuracy):
    critic_loss = np.hstack((np.loadtxt(save_critic_loss, delimiter=","),critic_loss))
    reward = np.hstack((np.loadtxt(save_reward, delimiter=",") ,total_rewards))
    accuracy =  np.hstack((np.loadtxt(save_accuracy, delimiter=",") ,accuracy))
    plt.plot(np.array(critic_loss), c='b', label='critic_loss(mean)',linewidth=0.5)
    plt.plot(np.array(reward), c='r', label='rewards(mean)',linewidth=0.5) #
    plt.plot(np.array(accuracy), c='m', label='accuracy*10',linewidth=0.6)
    plt.legend(loc=4)#
    plt.ylim(-2,12)
    #plt.ylim(-0.25,0.1)
    plt.ylabel('critic_loss') 
    plt.xlabel('training steps*200')
    plt.grid()
    plt.savefig(save_curve_pic)
    plt.close()
    np.savetxt(save_critic_loss,critic_loss,delimiter=',')
    np.savetxt(save_reward,reward,delimiter=',')
    np.savetxt(save_accuracy,accuracy,delimiter=',')

def initialization(list_agents,weight_path,agent_num):  #'/Tasks/A8_MAS_MNIST/weight/'
    if os.path.exists(weight_path + "actor0.pkl"):
        for i in range(agent_num):
            list_agents[i].actor.load_state_dict(torch.load(weight_path +"actor"+str(i)+".pkl"))
            list_agents[i].critic.load_state_dict(torch.load(weight_path +"critic"+str(i)+".pkl"))
        print('Model loaded')