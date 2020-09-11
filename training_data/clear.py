import os

def clear():

    f=open(os.path.abspath(os.curdir)+'/Tasks/A8_MAS_MNIST/training_data/loss.csv', "r+")
    f.truncate()
    f=open(os.path.abspath(os.curdir)+'/Tasks/A8_MAS_MNIST/training_data/reward.csv', "r+")
    f.truncate()
    f=open(os.path.abspath(os.curdir)+'/Tasks/A8_MAS_MNIST/training_data/accuracy.csv', "r+")
    f.truncate()
    
clear()