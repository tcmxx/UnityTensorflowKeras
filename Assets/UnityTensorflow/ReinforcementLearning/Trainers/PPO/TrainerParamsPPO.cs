using System.Collections;
using System.Collections.Generic;
using UnityEngine;

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
}
