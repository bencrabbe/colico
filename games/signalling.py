
import numpy as np
import numpy.random as R


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

        return {'alice': self.messages.copy(),
                'bob':self.actions.copy(),
                'reward':success}

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


if __name__ == '__main__':
    import visualize as V
    import matplotlib.pyplot as plt
    
    """
    Example signalling game with 3 nature states, 3 messages for Alice and 3 actions for Bob
    """

    G = SignallingGame(3,3,3)
    H = G.simulate_game(sample_size=1000)
    V.plot_alice_strategies(H)
    V.plot_bob_strategies(H)
    V.plot_reward(H)
    plt.show()
