import gym

from stable_baselines3.common.callbacks import BaseCallback

class CallbackActions(BaseCallback):
    def __init__(self, fakeactions, verbose=1):
        super(CallbackActions, self).__init__(verbose)
        print(id(fakeactions))
        self.fakeactions = fakeactions
        print(id(self.fakeactions))

    def _on_step(self) -> bool:
        #print('Called it')
        #print('Actions: ')
        #print(self.locals["actions"][0])

        #print('Env Action')
        #print(self.fakeactions["blockNr"])
        #print(self.fakeactions["blockPos"])
        self.locals["actions"][0][1] = self.fakeactions["blockNr"]
        self.locals["actions"][0][2] = self.fakeactions["blockPos"]

        #print('Dones: ')
        #print(self.locals["dones"])

        #print('Env')
        #env = self.locals["env"] 



        return True