using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using System.Text;
using System;

public class MaskInfoChannel : SideChannel
{


    public MaskInfoChannel()
    {
        ChannelId = new Guid("621f0a70-4f87-11ea-a6bf-784f4387d1aa");
    }
   
    protected override void OnMessageReceived(IncomingMessage msg)
    {
        //Do nothing
    }

    public void SendActionMsgToPython(Byte[] Actionmask)
    {
        using (var msgOut = new OutgoingMessage())
            {
                msgOut.SetRawBytes(Actionmask);
                QueueMessageToSend(msgOut);
                msgOut.Dispose();
            }
    }
}