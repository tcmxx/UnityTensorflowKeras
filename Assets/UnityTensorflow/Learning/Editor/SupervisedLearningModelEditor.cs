using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using UnityEditor;
using System.Linq;


[CustomEditor(typeof(SupervisedLearningModel))]
public class SupervisedLearningModelEditor : Editor
{

    public override void OnInspectorGUI()
    {
        base.OnInspectorGUI();
        SupervisedLearningModel model = (SupervisedLearningModel)target;
        EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);
        EditorGUILayout.LabelField("Network Parameters", GUI.skin.box);
        Editor.CreateEditor(model.network).OnInspectorGUI();
    }
}
