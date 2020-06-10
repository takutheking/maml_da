import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
import numpy as np
from data_generator import Data_Generator
from predictor import Predictor
import matplotlib.pyplot as plt
import copy
import math


def main():
    train_task_data_num,train_task_num=50,30
    meta_lr=0.001
    inner_lr=0.004
    data=Data_Generator(train_task_data_num,train_task_num)

    model=Predictor()
    optimizer=optim.SGD(model.parameters(),lr=meta_lr)
    criterion=nn.MSELoss()
    meta_epochs=18000
    update_num=1
    task_num=5
    time=[i for i in range(meta_epochs) ]
    time_losses=[0]*meta_epochs
    proposed=True
    function=0
    pre_num=0
    u=[1.0/math.sqrt(k+1) for k in range(meta_epochs)]
    init=[]
    for ele in model.state_dict().keys():
        init.append(model.state_dict()[ele])

    for epoch in range(meta_epochs):
        inputs, outputs, amp, phase=data.generate()
        losses=[0]*task_num
        for i in range(task_num):
            loss=F.mse_loss(torch.tensor(outputs[i][:train_task_data_num],dtype =torch.float32),model.forward(torch.tensor(inputs[i][:train_task_data_num],dtype = torch.float32)))
            grad = torch.autograd.grad(loss, model.parameters())
            fast_weights = list(map(lambda p: p[1] - inner_lr * p[0], zip(grad, model.parameters())))
            losses[i]=F.mse_loss(torch.tensor(outputs[i][train_task_data_num:],dtype=torch.float32),model.forward(torch.tensor(inputs[i][train_task_data_num:],dtype=torch.float32),vars=fast_weights))
        sum_losses=sum(losses)
        time_losses[epoch]=sum_losses.detach().numpy().copy().astype(float)
        if not proposed:
            optimizer.zero_grad()
            sum_losses.backward()
            optimizer.step()
        else:
            """
            grad = torch.autograd.grad(sum_losses, model.parameters())
            function+=(i+1.0+pre_num)/(i+2.0+pre_num)*function
            vars_temp=[]
            for ele in model.state_dict().keys():
                [[]]
                vars_temp.append(model.state_dict()[ele])
            difference=0
            difference2=0
            i=0
            for ele1,ele2 in zip(vars_temp,init):
                difference+=torch.sum(grad[i]*(ele1-ele2))
                difference2+=torch.sum((ele1-ele2)**2)
                i+=1

            real_function=function+1.0/(i+2.0)*difference+u[epoch]*difference2
            real_function.backward()
            optimizer.step()
        if epoch%1000==0:
            print(epoch)
            """



    test_task_data_num,test_task_num=30,1
    test_updates=10
    test_inputs, test_outputs, test_amp, test_phase=data.generate()
    for i in range(test_updates):
        optimizer.zero_grad()
        loss=criterion(torch.tensor(test_outputs[0][:test_task_data_num],dtype=torch.float32),model.forward(torch.tensor(test_inputs[0][:test_task_data_num],dtype=torch.float32)))
        loss.backward()
        optimizer.step()
    loss=criterion(torch.tensor(test_outputs[0][test_task_data_num:],dtype=torch.float32),model.forward(torch.tensor(test_inputs[0][test_task_data_num:],dtype=torch.float32)))
    x=np.linspace(-5.0,5.0,num=100)
    y_predict=model.forward(torch.tensor(x,dtype=torch.float32).reshape(100,1)).detach().numpy().reshape(-1)
    y=test_amp[0]*np.sin(x-test_phase[0])
    plt.plot(x,y)
    plt.plot(x,y_predict)
    plt.show()

if __name__=="__main__":
    main()
