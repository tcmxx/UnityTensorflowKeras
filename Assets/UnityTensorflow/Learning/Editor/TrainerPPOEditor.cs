using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using UnityEditor;
using System.Linq;


[CustomEditor(typeof(TrainerPPO))]
public class TrainerPPOEditor : Editor
{

    public override void OnInspectorGUI()
    {
        TrainerPPO myTrainer = (TrainerPPO)target;
        base.OnInspectorGUI();

        EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);
        EditorGUILayout.LabelField("Training Parameters", GUI.skin.box);
        myTrainer.parameters?.OnInspector();
        EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);

    }
}
