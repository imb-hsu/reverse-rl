using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using System.Text;
using System;

public class StepInfoChannel : SideChannel
{


    public StepInfoChannel()
    {
        ChannelId = new Guid("621f0a70-4f87-11ea-a6bf-784f4387d1f6");
    }

    protected override void OnMessageReceived(IncomingMessage msg)
    {
        var receivedString = msg.ReadString();
        Debug.Log("From Python : " + receivedString);
    }

    public void SendActionMsgToPython(int BlockConnectTo, int ConnectPos)
    {
        using (var msgOut = new OutgoingMessage())
            {
                msgOut.WriteInt32(BlockConnectTo);
                msgOut.WriteInt32(ConnectPos);
                QueueMessageToSend(msgOut);
                msgOut.Dispose();
            }
    }

    public void SendDebugStatementToPython(string logString, string stackTrace, LogType type)
    {
        if (type == LogType.Error)
        {
            var stringToSend = type.ToString() + ": " + logString + "\n" + stackTrace;
            using (var msgOut = new OutgoingMessage())
            {
                msgOut.WriteString(stringToSend);
                QueueMessageToSend(msgOut);
            }
        }
    }
}