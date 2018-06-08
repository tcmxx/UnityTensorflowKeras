using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using UnityEditor;
using System.Linq;


[CustomEditor(typeof(TrainerPPO))]
public class TrainerEditor : Editor
{

    public override void OnInspectorGUI()
    {
        TrainerPPO myBrain = (TrainerPPO)target;
        base.OnInspectorGUI();
        myBrain.parameters.OnInspector();
        
    }
}
