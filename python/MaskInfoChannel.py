from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
import numpy as np
import uuid


# Create the MaskInfoChannel class
class MaskInfoChannel(SideChannel):

    def __init__(self, actionmask) -> None:
        self.actionmask = actionmask
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1aa"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        self.actionmask[:] = msg.get_raw_bytes()

    def send_string(self, data: str) -> None:
        pass