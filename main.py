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
import sys


def main():
    if len(sys.argv) <2:
        print("Please type in which data to use: sinusoid || linear")
        sys.exit()
    #maml_hyperparameters
    meta_lr=0.001
    inner_lr=0.001
    update_num=1
    meta_epochs=1000
    plot_size=meta_epochs//100
    time=list(range(meta_epochs))
    time_losses=[0]*meta_epochs
    #data
    task_num,data_num=25,5
    data=Data_Generator(num_task=task_num,num_samples_per_task=data_num,task_type=sys.argv[1])
    #model
    model=Predictor()
    optimizer=optim.Adam(model.parameters(),lr=meta_lr)
    criterion=nn.MSELoss()
    #da_related
    proposed=True
    u=[0.00001/math.sqrt(k+1) for k in range(meta_epochs)]
    grad_list=[]
    linear_func=0
    #used for quadratic term
    init_parameters=[]
    for ele in model.state_dict().keys():
        init_parameters.append(model.state_dict()[ele])

    #beginning of the training
    for epoch in range(meta_epochs):
        if sys.argv[1]=="linear":
            pass
        elif sys.argv[1]=="sinusoid":
            inputs, outputs, amp, phase=data.generate()
        losses=[0]*task_num
        if epoch%plot_size==0:
            sys.stdout.write("\r%d" %epoch)
            sys.stdout.flush()
        for i in range(task_num):
            loss=F.mse_loss(torch.tensor(outputs[i][:data_num],dtype =torch.float32),model(torch.tensor(inputs[i][:data_num],dtype = torch.float32)))
            #if create_graph=False, first-order MAML?
            grad = torch.autograd.grad(loss, model.parameters(),create_graph=True)
            fast_weights = list(map(lambda p: p[1] - inner_lr * p[0], zip(grad, model.parameters())))
            losses[i]=F.mse_loss(torch.tensor(outputs[i][data_num:],dtype=torch.float32),model(torch.tensor(inputs[i][data_num:],dtype=torch.float32),vars=fast_weights))
        sum_losses=sum(losses)
        time_losses[epoch]=sum_losses.detach().numpy().copy().astype(float)
        if not proposed:
            optimizer.zero_grad()
            sum_losses.backward()
            optimizer.step()
        else:
            grad = torch.autograd.grad(sum_losses, model.parameters())
            "initializing"
            if epoch==0:
                grad_list=grad
            else:
                grad_list=list(map(lambda p: p[1]+p[0], zip(grad, grad_list)))
            quadratic=0
            for param,init in zip(model.parameters(),init_parameters):
                quadratic+=criterion(param,init)
            linear=0
            for param,g in zip(model.parameters(),grad_list):
                linear+=torch.sum(g*param)
            real_function=1.0/(epoch+1)*linear+u[epoch]*quadratic
            real_function.backward()
            optimizer.step()

    test_task_data_num,test_task_num=20,1
    test_updates=1
    test_inputs, test_outputs, test_amp, test_phase=data.generate()
    x=np.linspace(-5.0,5.0,num=100).reshape(-1,1)
    y_before=model(torch.tensor(x,dtype=torch.float32).reshape(-1,1)).detach().numpy()
    plt.plot(x,y_before,label="before finetuning")
    optimizer=optim.Adam(model.parameters(),lr=3*meta_lr)
    for i in range(test_updates):
        optimizer.zero_grad()
        loss=criterion(torch.tensor(test_outputs[0][:test_task_data_num],dtype=torch.float32),model(torch.tensor(test_inputs[0][:test_task_data_num],dtype=torch.float32)))
        loss.backward()
        optimizer.step()
    loss=criterion(torch.tensor(test_outputs[0][test_task_data_num:],dtype=torch.float32),model.forward(torch.tensor(test_inputs[0][test_task_data_num:],dtype=torch.float32)))
    y_predict=model(torch.tensor(x,dtype=torch.float32).reshape(-1,1)).detach().numpy()
    plt.plot(x,test_phase[0]*np.sin(x-test_amp[0]),label="oracle")
    plt.plot(x,y_predict,label="after fine-tuning")
    plt.legend()
    plt.show()

if __name__=="__main__":
    main()
