import numpy as np
import random
#originally taken from https://github.com/cbfinn/maml and modified
#modified from https://github.com/katerakelly/pytorch-maml as well.

class Data_Generator(object):
    def __init__(self,task_type,num_samples_per_task=25,num_task=5,config={}):
        self.num_task = num_task
        self.num_samples_per_task = num_samples_per_task
        if task_type=="linear":
            self.generate=self.generate_linear
        elif task_type=="sinusoid":
            self.generate=self.generate_sinusoid_batch
            self.amp_range = config.get('amp_range', [0.1, 5.0])
            self.phase_range = config.get('phase_range', [0, np.pi])
            self.input_range = config.get('input_range', [-5.0, 5.0])
            self.dim_input = 1
            self.dim_output = 1
        elif task_type=="omniglot":
            pass
        elif task_type=="miniimagenet":
            pass
        else:
            pass
    def generate_sinusoid_batch(self, train=True):
        amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.num_task]) #num_task is the number of tasks
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.num_task])
        outputs = np.zeros([self.num_task, self.num_samples_per_task*2, self.dim_output])
        inputs = np.zeros([self.num_task, self.num_samples_per_task*2, self.dim_input])
        for func in range(self.num_task):
            inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1], [self.num_samples_per_task*2, self.dim_input])
            outputs[func] = amp[func] * np.sin(init_inputs[func]-phase[func])
        return inputs, outputs, amp, phase

    def generate_linear(self,real_theta):
        self.theta_dim=len(real_theta)

        self.dim_input=self.theta_dim
        self.dim_output=1
        self.sigma_theta=s*np.eye(self.theta_dim)
        self.real_theta=real_theta
        theta=[]
        inputs=[]
        outputs=[]
        for i in range(self.num_task):
            theta_i=np.random.multivariate_normal(self.real_theta,self.sigma_theta)
            X_i=np.random.randn(self.theta_dim,self.num_samples_per_task)
            y_i=np.random.multivariate_normal(np.dot(theta_i,X_i),np.eye(self.num_samples_per_task))

            theta.append(theta_i)
            inputs.append(X_i)
            outputs.append(y_i)
        return theta,inputs,outputs
