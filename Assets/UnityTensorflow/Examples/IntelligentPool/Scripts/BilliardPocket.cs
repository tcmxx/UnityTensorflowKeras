/* Script added to the pocket triggers, just forward the OnTriggerEnter messages to Player
 * */

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BilliardPocket : MonoBehaviour {
    BilliardGameSystem gameSystem;
    public BilliardArena arena;
	// Use this for initialization
	void Start () {
        gameSystem = GameObject.FindObjectOfType<BilliardGameSystem>();
	}
    void OnTriggerEnter(Collider other)
    {
        gameSystem.onPocket(other.gameObject);
    }
}
