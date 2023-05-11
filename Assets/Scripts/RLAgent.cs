using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System;
using UnityEngine;
using UnityEngine.SceneManagement;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.SideChannels;

public class RLAgent : Agent
{
    [SerializeField]
    protected GameObject[] AllBlocks;
    [SerializeField]
    protected List<int[]> GoalConnections = new List<int[]>();
    [SerializeField]
    protected bool isReverse;
    [SerializeField]
    protected bool useVirtualActions;
    [SerializeField]
    protected bool useActionMask;
    [SerializeField]
    public int NUM_BLOCKS = 25;
    [SerializeField]
    protected MeshRenderer floorMesh;
    [SerializeField]
    protected Material winMat;
    [SerializeField]
    protected Material loseMat;
    StepInfoChannel stepChannel;
    MaskInfoChannel maskChannel;
    protected bool endNextStep = false; 

    public void Awake()
    {
        // We create the Side Channel
        stepChannel = new StepInfoChannel();
        maskChannel = new MaskInfoChannel();

        // When a Debug.Log message is created, we send it to the stringChannel
        Application.logMessageReceived += stepChannel.SendDebugStatementToPython;

        // The channel must be registered with the SideChannelManager class
        SideChannelManager.RegisterSideChannel(stepChannel);
        SideChannelManager.RegisterSideChannel(maskChannel);
    }

    public void OnDestroy()
    {
        // De-register the Debug.Log callback
        Application.logMessageReceived -= stepChannel.SendDebugStatementToPython;
        if (Academy.IsInitialized){
            SideChannelManager.UnregisterSideChannel(stepChannel);
            SideChannelManager.UnregisterSideChannel(maskChannel);
        }
    }


    public virtual void Start()
    {
        //Notation for block connections:
        // Position in list is the current block ID.
        // First param is the block its connected to, second is the connection position
        if(NUM_BLOCKS == 5){
            GoalConnections.Add( null );
            GoalConnections.Add(new int[]{0,2} );
            GoalConnections.Add(new int[]{0,6} );
            GoalConnections.Add(new int[]{1,3} );
            GoalConnections.Add(new int[]{2,5} );
        } 
        if(NUM_BLOCKS == 10){
            GoalConnections.Add( null );
            GoalConnections.Add(new int[]{0,2} );
            GoalConnections.Add(new int[]{0,6} );
            GoalConnections.Add(new int[]{1,3} );
            GoalConnections.Add(new int[]{2,5} );
            GoalConnections.Add(new int[]{3,1} );
            GoalConnections.Add(new int[]{3,6} );
            GoalConnections.Add(new int[]{4,6} );
            GoalConnections.Add(new int[]{6,1} );
            GoalConnections.Add(new int[]{7,2} );
        } 
        if(NUM_BLOCKS == 25){
            GoalConnections.Add( null );
            GoalConnections.Add(new int[]{0,1} );
            GoalConnections.Add(new int[]{0,5} );
            GoalConnections.Add(new int[]{1,1} );
            GoalConnections.Add(new int[]{1,5} );
            GoalConnections.Add(new int[]{2,5} );
            GoalConnections.Add(new int[]{3,1} );
            GoalConnections.Add(new int[]{3,5} );
            GoalConnections.Add(new int[]{4,5} );
            GoalConnections.Add(new int[]{5,5} );
            GoalConnections.Add(new int[]{6,1} );
            GoalConnections.Add(new int[]{7,1} );
            GoalConnections.Add(new int[]{8,1} );
            GoalConnections.Add(new int[]{9,1} );
            GoalConnections.Add(new int[]{9,5} );
            GoalConnections.Add(new int[]{10,3} );
            GoalConnections.Add(new int[]{12,3} );
            GoalConnections.Add(new int[]{14,3} );
            GoalConnections.Add(new int[]{15,5} );
            GoalConnections.Add(new int[]{16,1} );
            GoalConnections.Add(new int[]{16,5} );
            GoalConnections.Add(new int[]{17,1} );
            GoalConnections.Add(new int[]{18,5} );
            GoalConnections.Add(new int[]{19,5} );
            GoalConnections.Add(new int[]{20,5} );
        } 
    }

    //Is called at the start of every episode.
    public override void OnEpisodeBegin()
    {
        endNextStep = false;
        var envParameters = Academy.Instance.EnvironmentParameters;
        float get_reverse = envParameters.GetWithDefault("reverse", 0.0f);
        isReverse = Convert.ToBoolean(get_reverse);

        float get_virtual = envParameters.GetWithDefault("virtualactions", 0.0f);
        useVirtualActions = Convert.ToBoolean(get_virtual);

        
        float get_mask = envParameters.GetWithDefault("actionmask", 0.0f);
        useActionMask = Convert.ToBoolean(get_mask);

        //isReverse = true;
        Debug.Log("is Reverse");
        Debug.Log(isReverse);

        //useVirtualActions = true;
        Debug.Log("use virtual");
        Debug.Log(useVirtualActions);

        //useActionMask = true;
        Debug.Log("use mask");
        Debug.Log(useActionMask);

        //SceneManager.LoadScene("Basic");
        Debug.Log("New Scene Loaded");

        List<Vector3> AllPosRand = GenerateRandomStartingPos();
        if(isReverse == false)
        {
            //Generate the Starting Positions to enable Random Spots
            int i = 0;
            foreach(GameObject block in AllBlocks)
            {
                block.GetComponent<Rigidbody>().constraints = RigidbodyConstraints.FreezePositionZ | RigidbodyConstraints.FreezeRotationY
                    | RigidbodyConstraints.FreezeRotationX| RigidbodyConstraints.FreezeRotationZ | RigidbodyConstraints.FreezePositionX
                    | RigidbodyConstraints.FreezePositionY;
                
                FixedJoint fj = block.GetComponent<FixedJoint>();
                if(fj != null)
                { 
                    DestroyImmediate(fj);
                };
                block.transform.localPosition = AllPosRand[i];
                Block bl = block.GetComponent<Block>();
                bl.outOfBounds = false;
                bl.isConnected = false;
                bl.connectedBlockId = -1;
                bl.connectionPosition = -1;
                i++;
            }
        }
        else 
        {
            //Reverse Case
            int i = 0;
            foreach(GameObject block in AllBlocks)
            {
                block.GetComponent<Rigidbody>().constraints = RigidbodyConstraints.FreezePositionZ | RigidbodyConstraints.FreezeRotationY
                    | RigidbodyConstraints.FreezeRotationX| RigidbodyConstraints.FreezeRotationZ;
                
                
                block.SetActive(true);
                FixedJoint fj = block.GetComponent<FixedJoint>();
                if(fj != null)
                { 
                    DestroyImmediate(fj);
                };

                Block bl = block.GetComponent(typeof(Block)) as Block;
                bl.outOfBounds = false;
                bl.isConnected = false;
                bl.connectedBlockId = -1;
                bl.connectionPosition = -1;

                //Connect block to lower block, ignore lowest block.
                if(i == 0){
                    block.transform.position = AllPosRand[0];
                }
                
                if(GoalConnections[i] != null){
                    int conBlockId = GoalConnections[i][0];
                    int conBlockPos = GoalConnections[i][1];
                    Block conBlock = AllBlocks[conBlockId].GetComponent(typeof(Block)) as Block;
                    bl.ConnectWithBlock(conBlock, conBlockPos);
                }
                i++;
            }
        }

    }

    //Update Method, Check Out Of Bounds
    public void Update()
    {
        
        
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        foreach (GameObject block in AllBlocks)
        {
            Block bl = block.GetComponent(typeof(Block)) as Block;
            sensor.AddObservation(bl.isConnected);
            sensor.AddObservation(bl.connectedBlockId);
            sensor.AddObservation(bl.connectionPosition);
        }

        if(useActionMask){  
            byte[] actionMask = returnActionMask();
            maskChannel.SendActionMsgToPython(actionMask);
        }
    }


    public override void OnActionReceived(ActionBuffers actions)
    {
        //Actions
        int connectBlockID = actions.DiscreteActions[0];
        int connectWithBlockID = actions.DiscreteActions[1];
        int connectPosition = actions.DiscreteActions[2];
        //unsure if needed
        //if(useVirtualActions){
        //    stepChannel.SendActionMsgToPython(connectWithBlockID, connectPosition);
        //} 
        ConnectBlock(connectBlockID, connectWithBlockID, connectPosition);
        if(endNextStep){
             EndEpisode();
        }
        CheckIfCompleted();
    }

    public int getConnectedBlock(int BlockID){
        //Get ID of Block with which it is connected
        foreach(GameObject BlockGO in AllBlocks){
            Block Block = BlockGO.GetComponent(typeof(Block)) as Block;
            if(Block.connectedBlockId == BlockID){
                return Block.blockId;
            }
        }
        return -1;
    }

    public byte[] returnActionMask(){
        //Calculate the action mask for the current state
        byte[] actionMask = new byte[NUM_BLOCKS];
        int i = 0;
        foreach(GameObject blockGO in AllBlocks){
            Block block = blockGO.GetComponent(typeof(Block)) as Block;
            if(isReverse){
                actionMask[i] = Convert.ToByte(block.isConnected);
            }else{
                actionMask[i] = Convert.ToByte(!block.isConnected);
            }
            i++;
        }
        return actionMask;
    }

    public void ConnectBlock(int connectBlockID, int connectWithBlockID, int connectPosition){
        //Check if its standard or the reverse case
        Block connectBlock = AllBlocks[connectBlockID].GetComponent(typeof(Block)) as Block;
        Block connectWithBlock = AllBlocks[connectWithBlockID].GetComponent(typeof(Block)) as Block;

        if(!isReverse){
            //Block cannot connect with itself
            if(connectBlockID == connectWithBlockID){
                return;
            }
            //If the Block is already connected
            if(connectBlock.isConnected){
                return;
            }
            
            //Check if connection is correct according to goal
            if(GoalConnections[connectBlockID] == null){
                return;
            };
            if(GoalConnections[connectBlockID][0] != connectWithBlockID || 
                GoalConnections[connectBlockID][1] != connectPosition ) {
                return;
            };
     
            connectBlock.ConnectWithBlock(connectWithBlock, connectPosition);

            //Reward Shaping
            if(GoalConnections[connectBlockID] != null) {
                    if(GoalConnections[connectBlockID][0] == connectBlock.connectedBlockId &&
                    GoalConnections[connectBlockID][1] == connectBlock.connectionPosition ) {
                        SetReward(0.1f);
                    }
                }
        }else{
            //Dont try to disconnect if not connected
            if(!connectBlock.isConnected){
                //AddReward(-0.01f);
                print("Not Connected");
                print("Block ID " + connectBlockID + "Connect To Bl" + connectWithBlockID + " Pos " + connectPosition);
                return;
            }
            //Dont try to disconnect if the block is connected on both sides
            if(getConnectedBlock(connectBlockID) != -1 ){
                //AddReward(-0.01f);
                print("Both Sides");
                print("Block ID " + connectBlockID + "Connect To Bl" + connectWithBlockID + " Pos " + connectPosition);
                return;
            }
            
            print("Trying Disconnect");
            SetReward( 0.1f );
            //Sends the connction info to python, used as real disconnection
            if(useVirtualActions){
                stepChannel.SendActionMsgToPython(connectBlock.connectedBlockId, connectBlock.connectionPosition);
            } 
            connectBlock.DisconnectFromBlock(connectWithBlock, connectPosition);
        }

    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<int> discreteActions = actionsOut.DiscreteActions;

        discreteActions[0] = 0;
        if(Input.GetKey(KeyCode.V)){ discreteActions[0] = 0; }
        if(Input.GetKey(KeyCode.B)){ discreteActions[0] = 1; }
        if(Input.GetKey(KeyCode.N)){ discreteActions[0] = 2; }
        if(Input.GetKey(KeyCode.G)){ discreteActions[0] = 3; }
        if(Input.GetKey(KeyCode.H)){ discreteActions[0] = 4; }

        discreteActions[1] = 0;
        if(Input.GetKey(KeyCode.Y)){ discreteActions[1] = 0; }
        if(Input.GetKey(KeyCode.X)){ discreteActions[1] = 1; }
        if(Input.GetKey(KeyCode.C)){ discreteActions[1] = 2; }
        if(Input.GetKey(KeyCode.S)){ discreteActions[1] = 3; }
        if(Input.GetKey(KeyCode.D)){ discreteActions[1] = 4; }

        discreteActions[2] = 0;
        if(Input.GetKey(KeyCode.Alpha1)){ discreteActions[2] = 0; }
        if(Input.GetKey(KeyCode.Alpha2)){ discreteActions[2] = 1; }
        if(Input.GetKey(KeyCode.Alpha3)){ discreteActions[2] = 2; }
        if(Input.GetKey(KeyCode.Alpha4)){ discreteActions[2] = 3; }
        if(Input.GetKey(KeyCode.Alpha5)){ discreteActions[2] = 4; }
        if(Input.GetKey(KeyCode.Alpha6)){ discreteActions[2] = 5; }
        if(Input.GetKey(KeyCode.Alpha7)){ discreteActions[2] = 6; }

        print("Discrete Actions " + discreteActions[0] + " " + discreteActions[1] + " "  + discreteActions[2]);
    }


    //Check if all Blocks are Connected. End the Episode Successfully is they are.
    //Reverse Case: The opposite
    protected void CheckIfCompleted()
    {
        foreach(GameObject block in AllBlocks)
        {
            if(!isReverse){
                Block bl = block.GetComponent(typeof(Block)) as Block;
                //Check if the block is connected to the correct positions, if a connection is required
                if(GoalConnections[bl.blockId] != null) {
                    if(GoalConnections[bl.blockId][0] != bl.connectedBlockId ) {return;}
                    if(GoalConnections[bl.blockId][1] != bl.connectionPosition ) {return;}
                }
                //Check that the block is not connected if there is no goal connection
                if(GoalConnections[bl.blockId] == null) {
                    if(bl.isConnected){return;}
                }


            }else{
                Block bl = block.GetComponent(typeof(Block)) as Block;
                if(bl.isConnected == true) {return;}
            }
        }
        if(!isReverse){
            print("######Success: All Blocks Stacked ##########");
            floorMesh.material = winMat;
            //SetReward(1f);
            EndEpisode();
        }
        if(isReverse){
            print("######Success: All Blocks Disconnected ##########");
            floorMesh.material = winMat;
            //SetReward(1f);
            endNextStep = true; 
        }
        //SetReward(1f);
    }

    protected List<Vector3> GenerateRandomStartingPos()
    {
        Vector3 PosA = new Vector3(28, 0, 0);
        Vector3 PosB = new Vector3(24, 0, 0);
        Vector3 PosC = new Vector3(20, 0, 0);
        Vector3 PosD = new Vector3(16, 0, 0);
        Vector3 PosE = new Vector3(12, 0, 0);
        Vector3 PosF = new Vector3(28, 0, 20);
        Vector3 PosG = new Vector3(24, 0, 20);
        Vector3 PosH = new Vector3(20, 0, 20);
        Vector3 PosI = new Vector3(16, 0, 20);
        Vector3 PosJ = new Vector3(12, 0, 20);
        
        Vector3 PosK = new Vector3(28, 0, 10);
        Vector3 PosL = new Vector3(24, 0, 10);
        Vector3 PosM = new Vector3(20, 0, 10);
        Vector3 PosN = new Vector3(16, 0, 10);
        Vector3 PosO = new Vector3(12, 0, 10);
        Vector3 PosP = new Vector3(28, 0, -10);
        Vector3 PosQ = new Vector3(24, 0, -10);
        Vector3 PosR = new Vector3(20, 0, -10);
        Vector3 PosS = new Vector3(16, 0, -10);
        Vector3 PosT = new Vector3(12, 0, -10);
        Vector3 PosU = new Vector3(28, 0, -20);
        Vector3 PosV = new Vector3(24, 0, -20);
        Vector3 PosW = new Vector3(20, 0, -20);
        Vector3 PosX = new Vector3(16, 0, -20);
        Vector3 PosY = new Vector3(12, 0, -20);
        List<Vector3> AllPos = new List<Vector3>();
        AllPos.Add(PosA);
        AllPos.Add(PosB);
        AllPos.Add(PosC);
        AllPos.Add(PosD);
        AllPos.Add(PosE);
        AllPos.Add(PosF);
        AllPos.Add(PosG);
        AllPos.Add(PosH);
        AllPos.Add(PosI);
        AllPos.Add(PosJ);
        AllPos.Add(PosK);
        AllPos.Add(PosL);
        AllPos.Add(PosM);
        AllPos.Add(PosN);
        AllPos.Add(PosO);
        AllPos.Add(PosP);
        AllPos.Add(PosQ);
        AllPos.Add(PosR);
        AllPos.Add(PosS);
        AllPos.Add(PosT);
        AllPos.Add(PosU);
        AllPos.Add(PosV);
        AllPos.Add(PosW);
        AllPos.Add(PosX);
        AllPos.Add(PosY);

        System.Random rnd = new System.Random();
        List<Vector3> AllPosRand = AllPos.OrderBy(a => rnd.Next() ).ToList();
        return AllPosRand;
    } 
}

