using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using UnityEditor;
using System.Linq;


[CustomEditor(typeof(RLModelPPOHierarchy))]
public class RLModelPPOHierarchyEditor : Editor
{

    public override void OnInspectorGUI()
    {
        base.OnInspectorGUI();
        RLModelPPOHierarchy model = (RLModelPPOHierarchy)target;
        EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);
        EditorGUILayout.LabelField("Network Parameters", GUI.skin.box);
        Editor.CreateEditor(model.networkHierarchy).OnInspectorGUI();
    }
}
