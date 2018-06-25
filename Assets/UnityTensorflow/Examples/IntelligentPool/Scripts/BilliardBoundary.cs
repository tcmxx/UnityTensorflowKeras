using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BilliardBoundary : MonoBehaviour {

    public BilliardArena arena;

    void OnTriggerEnter(Collider other)
    {
        arena.OnOutOfBound(other.gameObject);
    }
}
