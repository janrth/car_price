# Create a Deep Averaging network model class
import torch


class TheModel(torch.nn.Module):
    def __init__(self,
                 act_fct,
                 inplace,
                 sizes):
        
        #super().__init__()
        super(TheModel, self).__init__()
        
        self.act_fct = act_fct
        self.inplace = inplace
        actns = [self.act_fct(inplace=self.inplace) for _ in range(len(sizes)-2)] + [None]

        layers = []
        for i,(n_in,n_out,act) in enumerate(zip(sizes[:-1],sizes[1:],actns)):
            layers.append(torch.nn.Linear(n_in, n_out))
            if act is not None: layers.append(act)
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, inputs):
        x = self.layers(inputs)
        return x # just return buffer for regression