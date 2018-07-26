using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;

[RequireComponent(typeof(DataPlane2D))]
[RequireComponent(typeof(EventTrigger))]
[RequireComponent(typeof(Collider))]
public class DataPlane2DInput : MonoBehaviour {


    protected DataPlane2D dataPlane;
    public int currentDataLabel;

    private void Awake()
    {
        dataPlane = GetComponent<DataPlane2D>();
    }
    // Use this for initialization
    void Start () {
		
	}
	
	// Update is called once per frame
	void Update () {
		
	}

    public void OnClicked(BaseEventData data)
    {
        var pdata = data as PointerEventData;
        var rcast = pdata.pointerCurrentRaycast;
        dataPlane.AddDatapoint(rcast.worldPosition, currentDataLabel);
        print("Clicked On " + rcast.worldPosition); 
    }

}
