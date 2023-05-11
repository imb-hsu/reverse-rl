from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
import numpy as np
import uuid


# Create the StepInfoChannel class
class StepInfoChannel(SideChannel):

    def __init__(self, fakeactions) -> None:
        self.fakeactions = fakeactions
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f6"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        #print('Block Nr')
        self.fakeactions["blockNr"] = msg.read_int32(default_value = 1000)
        #print(self.fakeactions["blockNr"])
        #print('Block Pos')
        self.fakeactions["blockPos"] =  msg.read_int32(default_value = 1001)
        #print(self.fakeactions["blockPos"])

    def send_string(self, data: str) -> None:
        # Add the string to an OutgoingMessage
        msg = OutgoingMessage()
        msg.write_string(data)
        # We call this method to queue the data we want to send
        super().queue_message_to_send(msg)