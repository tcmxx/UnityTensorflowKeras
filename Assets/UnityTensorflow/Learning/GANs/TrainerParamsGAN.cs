using System.Collections;
using System.Collections.Generic;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

[CreateAssetMenu()]
public class TrainerParamsGAN : TrainerParamsMimic
{
    [Header("GAN")]
    public int discriminatorTrainCount = 1;
    public int generatorTrainCount = 1;
    public bool usePrediction = false;
}
