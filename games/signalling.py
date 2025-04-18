
import numpy as np
import numpy.random as R
import matplotlib.pyplot as plt


def norm_weights(weights_vec):

    return weights_vec/weights_vec.sum()


# Roth Erev reinforcement
class SignallingGame:

    def __init__(self,nature_states,messages,actions):

        #uniform init (could be randomized)
        self.natureIDs  = nature_states 
        self.nature     = np.ones(nature_states)

        self.messagesIDs =  messages 
        self.messages   = np.ones((nature_states,messages))
        self.actionsIDs = actions
        self.actions    = np.ones((messages,actions))

    def snapshot(self,success=0):
        return ( self.messages.copy(),self.actions.copy(),success)

    def simulate_game(self,sample_size=100,update=True,Lambda=1.0):


        history = [self.snapshot()]
        for _ in range(sample_size):

            state   = R.choice(self.natureIDs,p=norm_weights(self.nature))
            message = R.choice(self.messagesIDs,p=norm_weights(self.messages[state]))
            action  = R.choice(self.actionsIDs,p=norm_weights(self.actions[message]))
            reward  = self.payoff(state,message,action)

            if update:
                self.messages[state][message] =  Lambda * self.messages[state][message] + reward
                self.actions[message][action]  = Lambda * self.actions[message][action] + reward

            history.append(self.snapshot(reward))

        return history


    def payoff(self,state,message,action):
        return int(state == action)


    def plot_alice_strategies(self,sim_history,normalize=True):

        messages  = [ M for (M,A,S) in sim_history ]
        
        F,axes = plt.subplots(1, self.natureIDs)
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
            axis.legend(  [f"m{mid}" for mid in range(self.messagesIDs) ])

            if normalize:
                axis.set_ylabel("Strategy probabilities")
            else:
                axis.set_ylabel("Strategy Weights")
          
    def plot_bob_strategies(self,sim_history,normalize=True):

        actions  = [ A for (M,A,S) in sim_history ]
        
        F,axes = plt.subplots(1, self.messagesIDs)
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
            axis.legend(  [f"a{mid}" for mid in range(self.actionsIDs) ])


    def plot_success(self,sim_history,sliding_window_size=100):
         
        succ  = [ S for (M,A,S) in sim_history ]
        avg   = [] 
        for idx in range(len(succ)):
            val = np.array(succ[idx-sliding_window_size:idx+1]).mean()    
            avg.append(val)

        plt.figure()
        plt.plot(avg)
        plt.title("Mean Success")
        plt.xlabel('Iteration')
        plt.ylabel('P(success)')


if __name__ == '__main__':
    G = SignallingGame(3,3,3)
    H = G.simulate_game(sample_size=1000)
    G.plot_alice_strategies(H)
    G.plot_bob_strategies(H)
    G.plot_success(H)
    plt.show()
