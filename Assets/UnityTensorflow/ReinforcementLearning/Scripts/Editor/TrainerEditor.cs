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

        EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);
        EditorGUILayout.LabelField("Training Parameters", GUI.skin.box);
        myBrain.parameters?.OnInspector();
        
    }
}
