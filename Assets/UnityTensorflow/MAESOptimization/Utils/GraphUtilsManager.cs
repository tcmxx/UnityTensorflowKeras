using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using AaltoGames;

public class GraphUtilsManager : MonoBehaviour {
    public Material materialZTestOff;
    public Material materialZTestOn;
    public DrawTime renderTime;
    public enum DrawTime
    {
        OnGui,
        OnRenderObject
    }
	// Use this for initialization
	void Start () {
        GraphUtils.setMaterials(materialZTestOff, materialZTestOn);
	}

    private void OnGUI()
    {
        if(renderTime == DrawTime.OnGui)
        {
            if (renderTime == DrawTime.OnRenderObject)
            {
                GraphUtils.DrawPendingLines();
            }
        }
    }
    // Update is called once per frame
    void OnRenderObject () {
        if (renderTime == DrawTime.OnRenderObject)
        {
            GraphUtils.DrawPendingLines();
        }
	}
    
}
