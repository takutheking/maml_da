import numpy as np
import random
import matplotlib.pyplot as plt


class maml_linear_regression:
    def __init__(self,task_num,data_num=10,theta_dim=10,random_state=0,theta_range=50,cov=False,epsilon=10,cov_const=30):
        self.task_num=task_num+1+1
        self.data_num=data_num
        self.random_state=random_state
        np.random.seed(random_state)
        self.epsilon=epsilon
        self.theta_dim=theta_dim
        self.real_theta=np.random.randint(-theta_range,theta_range,theta_dim).astype(np.float64)
        self.datas=[]
        for _ in range(self.task_num):
            #タスク毎のthetaを生成
            if not cov:
                sigma=cov_const*np.eye(theta_dim)
            else:
                sigma=np.random.randn(theta_dim,theta_dim)
            theta=np.random.multivariate_normal(self.real_theta,sigma)
            #データを生成
            if data_num:
                X=np.random.randn(data_num,theta_dim)
                y=np.random.multivariate_normal(np.dot(X,theta),epsilon*np.eye(data_num))
            else:
                temp=random.randrange(10,1000)
                X=np.random.randn(temp,theta_dim)
                y=np.random.multivariate_normal(np.dot(X,theta),epsilon*np.eye(temp))
            self.datas.append((X,y,theta))
    def gradient(self,X,y,theta):
        return 2*np.dot(np.dot(X.T,X),theta)-2*np.dot(X.T,y).reshape(-1,1)
    def meta_gradient(self,X,y,theta,alpha=0.001,beta=0.01,grad_fixed=True,step=1,mini=False,batch_size=10):
        def pow(m,nth):
            if nth==1:
                return m
            else:
                return np.dot(m,pow(m,nth-1))
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

    def fit(self,random_theta=False,theta_range=50,epochs=100000,early_stopping=100,eta=0.001,datas=False,outputs=False):
        theta=np.random.randint(-theta_range,theta_range,self.theta_dim).astype(np.float64).reshape(-1,1) #初期値
        i=0 #early stoppingからの回数
        if not datas or not outputs:
            X_val,y_val,theta_v=self.datas[len(self.datas)-2]
            X_test,y_test,theta_test=self.datas[len(self.datas)-1]
        else:
            X_val,
        theta_val=theta
        self.process=[]
        self.test_process=[]
        self.real_process=[]
        best=1000000000
        for  in range(epochs):

            theta+=self.meta_batch_update(theta)

            theta_val=theta-eta*self.gradient(X_val,y_val,theta)
            ans=np.sum((theta_val-theta_v.reshape(-1,1))**2)
            self.process.append(ans)
            self.test_process.append(np.sum((theta-eta*self.gradient(X_test,y_test,theta)-theta_test.reshape(-1,1))**2))
            self.real_process.append(np.sum((self.real_theta-theta_test)**2))
            if best<ans:
                i+=1
                if early_stopping<=i:
                    break
            else:
                i=0
                best=ans
        return theta-eta*self.gradient(X_test,y_test,theta)
