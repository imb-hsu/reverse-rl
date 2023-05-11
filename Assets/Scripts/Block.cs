using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/*
This Script governs the behaviour of a single block.
 
*/

public class Block : MonoBehaviour
{
    public bool outOfBounds = false;

    //This block connection
    public bool isConnected = false;
    public int connectedBlockId = -1;
    public int connectionPosition = -1;

    public string color;
    public int blockId;  

    public Transform knob11;
    public Transform knob12;
    public Transform knob13;
    public Transform knob14;
    public Transform knob21;
    public Transform knob22;
    public Transform knob23;
    public Transform knob24;
    
    //Check if the block is touching another block
    private void OnTriggerEnter(Collider other)
    {
        if (other.TryGetComponent<OutOfBounds>(out OutOfBounds oob ) )
        {
            outOfBounds = true;
        }
    }

    //Disconnect function to mirror the connect function lower.
    //Optional position to specify which position it was connected
    public bool DisconnectFromBlock(Block Block, int position){

        //Current Block gets released
        
        if(isConnected){
            FixedJoint fj = gameObject.GetComponent<FixedJoint>() as FixedJoint;
            Destroy(fj);
            isConnected = false;
            connectedBlockId = -1;
            connectionPosition = -1;
            gameObject.SetActive(false);
            return true;
        }
        return false;
    }


    //Connect directly with a block, no checks taken
    public bool ConnectWithBlock(Block Block, int position)
    {
            FixedJoint fj = gameObject.AddComponent(typeof( FixedJoint ) ) as FixedJoint;
            Rigidbody rb = Block.GetComponent(typeof( Rigidbody ) ) as Rigidbody;

            //Set the positin on top of the other block. Expressed in local Coordinates, adjusted to scale!
            gameObject.transform.rotation = Block.transform.rotation;

            switch(position)
            {
                case 0:
                gameObject.transform.position = Block.knob11.TransformPoint( new Vector3(0.024f, 0.0f, 0f) );
                break;

                case 1:
                gameObject.transform.position = Block.knob11.TransformPoint( new Vector3(0.016f, 0.0f, 0f) );
                break;

                case 2:
                gameObject.transform.position = Block.knob11.TransformPoint( new Vector3(0.008f, 0.0f, 0f) );
                break;

                case 3:
                gameObject.transform.position = Block.knob11.TransformPoint( new Vector3(0.0f, 0.0f, 0f) );
                break;

                case 4:
                gameObject.transform.position = Block.knob12.TransformPoint( new Vector3(0.0f, 0.0f, 0f) );
                break;

                case 5:
                gameObject.transform.position = Block.knob13.TransformPoint( new Vector3(0.0f, 0.0f, 0f) );
                break;

                case 6:
                gameObject.transform.position = Block.knob14.TransformPoint( new Vector3(0.0f, 0.0f, 0f) );
                break;
            }
            
            connectedBlockId = Block.blockId;
            connectionPosition = position;

            fj.connectedBody = rb;
            Debug.Log("Connected");
            isConnected = true;

            return true;
    }

}
