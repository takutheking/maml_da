import torch
from torch import nn
import torch.nn.functional as F

class Predictor(nn.Module):
    def __init__(self,config=[("linear",(40,1)),("relu",(40)),("linear",(40,40)),("relu",(40)),("linear",(1,40))]):
        super(Predictor,self).__init__()
        self.config=config
        self.vars=nn.ParameterList()
        for i,(name,param) in enumerate(self.config):
            if name is "linear":
                w=nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            if name in ["relu"]:
                continue
            else:
                pass

    def forward(self,x,vars=None):
        if vars is None:
            vars=self.vars
        idx=0
        for name,param in self.config:
            if name is "linear":
                w,b=vars[idx],vars[idx+1]
                x=F.linear(x,w,b)
                idx+=2
            elif name in ["relu"]:
                x=F.relu(x)

        assert idx==len(vars)
        return x
    def parameters(self):
        return self.vars
