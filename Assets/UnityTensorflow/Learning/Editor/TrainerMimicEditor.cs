using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using UnityEditor;
using System.Linq;


[CustomEditor(typeof(TrainerMimic))]
public class TrainerMimicEditor : Editor
{

    public override void OnInspectorGUI()
    {
        TrainerMimic myTrainer = (TrainerMimic)target;
        base.OnInspectorGUI();

        EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);
        EditorGUILayout.LabelField("Training Parameters", GUI.skin.box);
        myTrainer.parameters?.OnInspector();
        EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);

    }
}
