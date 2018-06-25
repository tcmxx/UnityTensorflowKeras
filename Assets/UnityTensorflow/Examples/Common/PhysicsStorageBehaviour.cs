using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PhysicsStorageBehaviour : MonoBehaviour {





    public class RigidbodyState
    {
        public Rigidbody b;
        public Vector3 pos, vel, aVel;
        public Quaternion q;
        public bool active;
        public bool isSleeping;
    }
    List<RigidbodyState> bodyStates = new List<RigidbodyState>();


    public void StopAll()
    {
        Rigidbody[] bodies = GetComponentsInChildren<Rigidbody>(true);
        foreach (Rigidbody b in bodies)
        {
            b.velocity = Vector3.zero;
        }
    }

    public bool IsAllSleeping()
    {
        Rigidbody[] bodies = GetComponentsInChildren<Rigidbody>(true);
        foreach(var b in bodies)
        {
            if (!b.IsSleeping())
                return false;
        }
        return true;
    }

    public List<RigidbodyState> SaveState()
    {
        Rigidbody[] bodies = GetComponentsInChildren<Rigidbody>(true);
        bodyStates = new List<RigidbodyState>();
        foreach (Rigidbody b in bodies)
        {
            RigidbodyState state = new RigidbodyState();
            state.b = b;
            state.pos = b.position;
            state.vel = b.velocity;
            state.aVel = b.angularVelocity;
            state.q = b.rotation;
            state.active = b.gameObject.activeSelf;
            state.isSleeping = b.IsSleeping();
            bodyStates.Add(state);
        }
        return bodyStates;
    }
    public void RestoreState(List<RigidbodyState> states = null)
    {
        if(states != null)
        {
            bodyStates = states;
        }
        foreach (RigidbodyState state in bodyStates)
        {
            state.b.gameObject.SetActive(state.active);
            state.b.position = state.pos;
            state.b.velocity = state.vel;
            state.b.angularVelocity = state.aVel;
            state.b.rotation = state.q;

            state.b.transform.position = state.pos;
            state.b.transform.rotation = state.q;
            if (state.isSleeping)
            {
                state.b.Sleep();
            }
        }
    }
    //Use this if you want restoreState() have immediate effect on Unity's transforms. Otherwise, they only get updated after the next physics step.
    public void RestoreTransforms()
    {
        foreach (RigidbodyState state in bodyStates)
        {
            state.b.transform.position = state.pos;
            state.b.transform.rotation = state.q;
        }
    }
}
