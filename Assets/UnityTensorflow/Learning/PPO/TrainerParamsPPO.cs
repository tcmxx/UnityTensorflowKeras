using System.Collections;
using System.Collections.Generic;
using UnityEngine;


[CreateAssetMenu()]
public class TrainerParamsPPO : TrainerParams
{
    [Header("Learning related")]

    public float rewardDiscountFactor = 0.99f;
    public float rewardGAEFactor = 0.95f;
    public float valueLossWeight = 1f;
    public int timeHorizon = 1000;
    [Tooltip("larger value means exploration is encouraged")]
    public float entropyLossWeight = 0.0f;
    public float clipEpsilon = 0.2f;

    
    public int batchSize = 128;
    public int bufferSizeForTrain = 2048;
    public int numEpochPerTrain = 10;

    public int heuristicBufferSize = 0;
    public int extraBatchTFromHeuristicBuffer = 0;
    
    [Header("Log related")]
    
    public int rewardLogInterval = 10;

}
