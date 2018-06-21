using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;

public class BilliardHeatMapToParam : MonoBehaviour {

    public BilliardGameSystem gameSystemRef;
    public BilliardAgent agentRef;
    public BilliardSimple agentSimplRef;

    public bool isSampling { get; set; }

    protected Vector3 sampledForce;
	// Use this for initialization
	void Start () {
		
	}
	
	// Update is called once per frame
	void Update () {
        if (isSampling) {
            RaycastHit hit;
            if(Physics.Raycast(Camera.main.ScreenPointToRay(Input.mousePosition),out hit))
            {
                if (hit.collider.gameObject.name == "HeatMap")
                {
                    Vector3 relatedPoint = transform.InverseTransformPoint(hit.point);
                    if(agentRef != null)
                        sampledForce = agentRef.SamplePointToForceVectorXY(relatedPoint.x + 0.5f, relatedPoint.y + 0.5f);
                    else if(agentSimplRef != null)
                    {
                        sampledForce = agentSimplRef.SamplePointToForceVectorXY(relatedPoint.x + 0.5f, relatedPoint.y + 0.5f);
                    }
                    gameSystemRef.evaluateShot(sampledForce, Color.green);
                }
            }
        }
	}


    public void Shoot()
    {
        gameSystemRef.shoot(sampledForce);
    }
    

}
