import numpy as np
import random
import matplotlib.pyplot as plt


class maml_linear_regression:
    def __init__(self,num_task=10,num_data=25,dim=10):
        self.num_task=num_task
        self.num_data=num_data
        data=Data_Generator(task_type="linear",num_samples_per_task=self.num_data,num_task=self.num_task)
        self.theta_dim=dim
        self.real_theta=np.random.randn(self.theta_dim)
        #(num_task,theta_dim), (num_task,2*num_data,theta_dim), (num_task,2*num_data)
        self.theta,self.inputs,self.outputs=data.generate(self.real_theta)

    def gradient(self,X,y,theta): #gradient of loss function by theta.
        return 2*np.dot(np.dot(X.T,X),theta)-2*np.dot(X.T,y)
    def meta_gradient(self,X,y,theta,alpha=0.001,beta=0.01,grad_fixed=True,step=1,mini=False,batch_size=10):
        accum=np.zeros((self.theta_dim,self.theta_dim))
        if mini:
            zipped=np.array(zip(X,y))
            np.random.permutation(zipped)
            X,y=zip(*zipped)
            X=np.array(X[:batch_size])
            y=np.array(y[:batch_size])
        k=np.dot(X.T,X)
        z=np.dot(X.T,y)
        temp=np.eye(self.theta_dim)-2*alpha*k
        if grad_fixed:
            #exact version of gradient
            accum=np.dot(np.dot(temp,temp),np.dot(k,theta)-z.reshape(-1,1))
        else:
            #approximate version of gradient
            accum=np.dot(temp,np.dot(k,theta)-z.reshape(-1,1))
        return (-2*beta*accum).reshape(-1,1)


    def meta_batch_update(self,theta,alpha=0.001,beta=0.0001,grad_fixed=True,step=1,mini=False,batch_size=10,meta_mini=False):
        accum=np.zeros((self.theta_dim,1))
        if not meta_mini:
            for data in self.datas[:len(self.datas)-2]:
                X,y,_=data
                accum=accum+np.array(self.meta_gradient(X,y,theta,alpha=alpha,beta=beta,grad_fixed=grad_fixed,step=step,mini=mini,batch_size=batch_size))
        else:
            num=np.random.randint(1,len(self.datas))
            datas=self.datas
            np.random.permutation(datas)
            for data in datas[:num]:
                X,y,_=data
                accum=accum+np.array(self.meta_gradient(X,y,theta,alpha=alpha,beta=beta,grad_fixed=grad_fixed,step=step,mini=mini,batch_size=batch_size)).reshape(-1,1)
        return accum

    def fit(self,epochs=10000,early_stopping=100,lr=0.001,datas=False,outputs=False):
        parameters=np.random.randn(self.theta_dim) #初期値
        early_stopping=0 #early stoppingからの回数
        self.process=[]
        self.test_process=[]
        self.real_process=[]
        best=1000000000

        for epoch in range(epochs):
            parameter+=self.meta_batch_update(parameter)

            theta_val=parameter-lr*self.gradient(X_val,y_val,parameter)
            ans=np.sum((theta_val-theta_v.reshape(-1,1))**2)
            self.process.append(ans)
            self.test_process.append(np.sum((parameter-lr*self.gradient(X_test,y_test,parameter)-theta_test.reshape(-1,1))**2))
            self.real_process.append(np.sum((self.real_theta-theta_test)**2))
            if best<ans:
                early_stopping+=1
                if early_stopping<=early_stopping:
                    break
            else:
                early_stopping=0
                best=ans
        return parameter-lr*self.gradient(X_test,y_test,parameter)
