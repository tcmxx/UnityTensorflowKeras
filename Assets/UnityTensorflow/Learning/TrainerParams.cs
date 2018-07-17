using System.Collections;
using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

public class TrainerParams : ScriptableObject
{
    public float learningRate = 0.001f;
    public int maxTotalSteps = 100000000;
    public int lossLogInterval = 1;
    public int saveModelInterval = 10000;
    /// Displays the parameters of the CoreBrainInternal in the Inspector 
    public virtual void OnInspector()
    {
#if UNITY_EDITOR
        Editor.CreateEditor(this).OnInspectorGUI();
#endif
    }
}
