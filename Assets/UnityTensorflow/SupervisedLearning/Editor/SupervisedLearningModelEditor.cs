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
        SupervisedLearningModel model = (SupervisedLearningModel)target;
        base.OnInspectorGUI();

        EditorGUILayout.LabelField("",GUI.skin.horizontalSlider);
        EditorGUILayout.LabelField("Network Parameters", GUI.skin.box);
        model.network?.OnInspector();
    }
}
