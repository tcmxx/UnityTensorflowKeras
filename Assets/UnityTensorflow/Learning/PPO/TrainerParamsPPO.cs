using System.Collections;
using System.Collections.Generic;
using UnityEngine;


[CreateAssetMenu(menuName = "ml-agent/ppo/TrainerParamsPPO")]
public class TrainerParamsPPO : TrainerParams
{
    [Header("Learning related")]
    [Tooltip("gamma")]
    public float rewardDiscountFactor = 0.99f;
    [Tooltip("lambda")]
    public float rewardGAEFactor = 0.95f;
    public float valueLossWeight = 0.5f;
    public int timeHorizon = 1000;
    [Tooltip("larger value means exploration is encouraged")]
    public float entropyLossWeight = 0.01f;
    public float clipEpsilon = 0.2f;
    public float clipValueLoss = 0.2f;

    public int batchSize = 128;
    public int bufferSizeForTrain = 2048;
    public int numEpochPerTrain = 3;

    //[Range(0,1)]
    //public float useHeuristicChance = 0.4f;

    [Tooltip(" Unity's impelemntation does clip the normalize the final acion for continuous space before sending to agents: clip(action,-3,3)/3.")]
    public float finalActionClip = 3;
    [Tooltip(" Unity's impelemntation does clip the normalize the final acion for continuous space before sending to agents: clip(action,-3,3)/3.")]
    public float finalActionDownscale = 3;

}
