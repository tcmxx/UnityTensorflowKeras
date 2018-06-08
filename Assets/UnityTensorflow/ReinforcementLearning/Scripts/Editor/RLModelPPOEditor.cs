using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using UnityEditor;
using System.Linq;


[CustomEditor(typeof(RLModelPPO))]
public class RLModelEditor : Editor
{

    public override void OnInspectorGUI()
    {
        RLModelPPO model = (RLModelPPO)target;
        base.OnInspectorGUI();

        EditorGUILayout.LabelField("",GUI.skin.horizontalSlider);
        EditorGUILayout.LabelField("Network Parameters", GUI.skin.box);
        model.network?.OnInspector();
    }
}
