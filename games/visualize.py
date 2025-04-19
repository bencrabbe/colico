import numpy as np
import matplotlib.pyplot as plt

"""
This file provides example plotting functionalities for plotting metrics from simulations
The history data structure is a list of dictionaries (one dictionary by simulation step)

There is at least an "alice", a "bob" and a "reward" field in each dictionary
* The 'alice' field contains a numpy matrix (states x strategies) with Alice messaging strategies given each state
* The 'bob' field contains a numpy matrix (states x strategies) with Bob action strategies given each state
* The 'reward' field is an integer the reward received 
"""

def plot_alice_strategies(sim_history,normalize=True):
        """
        Plotting for the Alice agent:
        Args:
            sim_history (list of dict): list of dicts as described above. An alice matrix is a numpy array (states x strategies)
        KwArgs:
            normalize (bool): whether weights are normalized to probabilities or not (weights are supposed to be positive integers)
        """
        messages  = [ dataitem['alice'] for dataitem in sim_history ]
        
        nstates,nmessages = messages[-1].shape

        F,axes = plt.subplots(1, nstates)
        F.suptitle("Messaging strategies given Nature state")

        print("Alice")
        print(messages[-1])

        for state_idx,axis in enumerate(axes):
            
            state_strategies =  np.stack([ M[state_idx] for M in messages ])
            if normalize:
                state_strategies /= state_strategies.sum(axis=1,keepdims=True)


            axis.plot(state_strategies)             
            axis.set_xlabel("iterations")
            axis.set_title(f'State {state_idx}')
            axis.legend(  [f"m{mid}" for mid in range(nmessages) ])

            if normalize:
                axis.set_ylabel("Strategy probabilities")
            else:
                axis.set_ylabel("Strategy Weights")


def plot_bob_strategies(sim_history,normalize=True):
        """
        Plotting for the Bob agent:
        Args:
            sim_history (list of dict): list of dicts as described above. An alice matrix is a numpy array (states x strategies)
        KwArgs:
            normalize (bool): whether weights are normalized to probabilities or not (weights are supposed to be positive integers)
        """

        actions  = [ dataitem['bob'] for dataitem in sim_history ]
        

        nstates,nactions = actions[-1].shape

        F,axes = plt.subplots(1, nstates)
        F.suptitle("Action strategies given Messaging state")
        

        print("Bob")
        print(actions[-1])

        for state_idx,axis in enumerate(axes):
            message_strategies =  np.stack([ A[state_idx] for A in actions ])
            if normalize:
                message_strategies /= message_strategies.sum(axis=1,keepdims=True)
            axis.plot(message_strategies)             
            axis.set_xlabel("iterations")
            if normalize:
                axis.set_ylabel("Strategy probabilities")
            else:
                axis.set_ylabel("Strategy weights")
            axis.set_title(f'State {state_idx}')
            axis.legend(  [f"a{mid}" for mid in range(nactions) ])


def plot_reward(sim_history,sliding_window_size=100):
        """
        Plotting for the Reward:
        Args:
            sim_history (list of dict): list of dicts as described above. The reward is a number
        KwArgs:
            sliding_window_size (int): the plot uses a sliding window to smooth the reward plot
        """         

        succ  = [ dataitem['reward'] for dataitem in sim_history ]
        avg   = [] 
        for idx in range(len(succ)):
            val = np.array(succ[idx-sliding_window_size:idx+1]).mean()    
            avg.append(val)

        plt.figure()
        plt.plot(avg)
        plt.title("Mean Reward")
        plt.xlabel('Iteration')
        plt.ylabel('Reward')

