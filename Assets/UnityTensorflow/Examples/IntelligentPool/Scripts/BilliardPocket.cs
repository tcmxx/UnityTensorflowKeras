/* Script added to the pocket triggers, just forward the OnTriggerEnter messages to Player
 * */

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BilliardPocket : MonoBehaviour {
    public BilliardArena arena;

    void OnTriggerEnter(Collider other)
    {
        arena.OnPocket(other.gameObject);
    }
}
