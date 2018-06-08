using System.Collections;
using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

[CreateAssetMenu()]
public class TrainerParamsPPO : ScriptableObject {

    public int maxTotalSteps = 100000000;

    public float rewardDiscountFactor = 0.99f;
    public float rewardGAEFactor = 0.95f;
    public float valueLossWeight = 1f;
    public float entroyLossWeight = 0.0f;
    public float clipEpsilon = 0.2f;

    public int batchSize = 128;
    public int bufferSizeForTrain = 2048;
    public int numEpochPerTrain = 100;

    public float learningRate = 0.001f;

    /// Displays the parameters of the CoreBrainInternal in the Inspector 
    public void OnInspector()
    {
#if UNITY_EDITOR
        Editor.CreateEditor(this).OnInspectorGUI();
#endif
    }
}
