
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AutoTrainAndLog : MonoBehaviour
{


    public Trainer trainerRef;
    public string sessionName;
    public string directoryName;
    public int index = 1;
    // Use this for initialization
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        int currentStep = trainerRef.GetStep();
        int maxStep = trainerRef.GetMaxStep();

        if (currentStep >= maxStep)
        {
            Grapher.SaveToFiles(sessionName + "_" + index, directoryName);
            Grapher.Reset();
            trainerRef.ResetTrainer();
            KerasSharp.Backends.Current.K.try_initialize_variables(false);
            index++;
        }
    }
}
