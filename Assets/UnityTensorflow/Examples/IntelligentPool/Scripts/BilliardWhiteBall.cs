using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BilliardWhiteBall : MonoBehaviour {


    public bool TouchedOtherBall { get; set; }
    private void OnCollisionEnter(Collision collision)
    {
        if(collision.rigidbody != null)
        {
            TouchedOtherBall = true;
        }
    }
}
