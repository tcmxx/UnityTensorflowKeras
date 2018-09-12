using System.Collections;
using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

public class TrainerParams : ScriptableObject
{
    [Header("Basic Parameters")]
    public float learningRate = 0.001f;
    public int maxTotalSteps = 100000000;
    public int saveModelInterval = 10000;
    [Header("Log related")]
    public int logInterval = 1000;
}
