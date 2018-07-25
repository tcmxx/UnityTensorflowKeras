using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BilliardTableEdge : MonoBehaviour {
    public BilliardArena arena;
    // Use this for initialization
    void Start () {
		
	}

    void OnCollisionEnter(Collision col)
    {
        arena.OnBounceEdge(col.gameObject);
    }
}
