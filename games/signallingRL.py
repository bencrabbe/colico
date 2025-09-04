import torch
import torch.nn as nn
from torch.nn import LogSoftmax,Softmax
from torch.optim import AdamW
from torch.distributions import Categorical

"""
This is an implementation of a signalling game with standard torch implementation.
"""
class DiscreteStateAgent(nn.Module):
    """
    An agent performing actions given discrete states
    """
    def __init__(self,nstates,nactions):
        
        super().__init__()
        self.params        = nn.Embedding(nactions,nstates)
        self.logsoftmax    = LogSoftmax()
    
    def forward(self,state_idx):
        """
        This return the actions log probabilities P(A|state_idx)
        Args:
            state_idx (int): the integer id for the state
        """
        return self.logsoftmax(self.params(state_idx))

    def snapshot(self,use_softmax=True):
        """
        Returns a numpy matrix with a copy of the weights

        KwArgs:
            use_softmax (bool): if true, the returned weights are softmaxed rowwise 
        """
        W = self.params.weight.detach()
        if use_softmax:     
            softmax = nn.Softmax(dim=0)
            return softmax(W).numpy()        
        return W.numpy()



class SignallingGame:

    def __init__(self,nstates,alice,bob):
        """
        The Signalling game environment taking two discrete agents Alice and Bob

        Args:
            nstates              (int) : the number of states sampled by Nature
            alice (DiscreteStateAgent) : agent A
            bob   (DiscreteStateAgent) : agent B
        """
        #defaults to uniform sampling for Nature
        self.nstates        = nstates
        self.nature_sampler = Categorical(torch.ones(nstates)/nstates)
        self.alice          = alice
        self.bob            = bob


    def sample_nature(self):
        idx = self.nature_sampler.sample()
        return idx,self.nature_sampler.log_prob(idx)


    def sample_agent(self,agent,state):

        logits  = agent(state).float()
        sampler = Categorical(logits=logits)
        idx     = sampler.sample()
        return (idx,sampler.log_prob(idx))


    def reward(self,a,b):

        return 1 if a == b else -1
     

    def sample_episode(self,ntrajectories=4):
        """
        Samples ntrajectories and computes the loss
        """
        loss_lst = [ ]  
        reward_lst = [ ] 
        for _ in range(ntrajectories):
            nat_state,nat_logp = self.sample_nature()
            msg_state,msg_logp = self.sample_agent(self.alice,nat_state)
            act_state,act_logp = self.sample_agent(self.bob,msg_state)
            reward             = self.reward(nat_state,act_state)
            loss_lst.append(reward * (act_logp + msg_logp))
            reward_lst.append(reward)

        loss   = -sum(loss_lst) / ntrajectories #turns the problem into a maximization problem
        reward = sum(reward_lst)/ntrajectories

        print("Expected Reward",reward)
        return loss,reward
            

    def sample_game(self,niter=200,ntrajectories=16):
        """
        Trains a reinforcement model 
        """
        optim_fnc = AdamW( list(self.alice.parameters()) + list(self.bob.parameters()),lr=0.1)

        history = []

        for step in range(niter):
            optim_fnc.zero_grad()
            loss,reward = self.sample_episode(ntrajectories)
            loss.backward()
            print('loss',loss.item())
            optim_fnc.step()


            history.append({'alice':self.alice.snapshot(),'bob':self.bob.snapshot(),'reward':reward})


        return history






if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt    
    import visualize as V

    Alice = DiscreteStateAgent(3,3)
    Bob   = DiscreteStateAgent(3,3)

    G = SignallingGame(3,Alice,Bob)
    H = G.sample_game()
    V.plot_alice_strategies(H,normalize=False)
    V.plot_bob_strategies(H,normalize=False)
    V.plot_reward(H)

    plt.show()
