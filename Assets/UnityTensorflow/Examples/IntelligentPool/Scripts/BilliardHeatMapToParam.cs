using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

public class BilliardHeatMapToParam : MonoBehaviour {

    public BilliardGameSystem gameSystemRef;
    public BilliardAgent agentRef;
    public BilliardSimple agentSimplRef;
    public LayerMask heatmapLayer;

    public Text heatmapInfo;

    public bool IsSampling { get {
            return isSampling;
        } set {
            
            Physics.autoSimulation = !value;
            isSampling = value;
            //print("Is samplling:" + isSampling);
        } }
    protected bool isSampling;
    protected Vector3 sampledForce;
	// Use this for initialization
	void Start () {
		
	}
	
	// Update is called once per frame
	void Update () {
        if (isSampling) {
            RaycastHit hit;
            if(Physics.Raycast(Camera.main.ScreenPointToRay(Input.mousePosition),out hit, heatmapLayer))
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
                    float score = gameSystemRef.EvaluateShot(sampledForce, Color.green);

                    heatmapInfo.text = sampledForce.x + ", " + sampledForce.z + ": " + score;
                }
            }
        }
	}


    public void Shoot()
    {
        Physics.autoSimulation = true;
        gameSystemRef.Shoot(sampledForce);
    }
    

}
