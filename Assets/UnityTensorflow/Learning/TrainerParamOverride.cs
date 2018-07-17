using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(Trainer))]
public class TrainerParamOverride : MonoBehaviour {

    

    public List<FieldOverride> overrides = new List<FieldOverride>();

    [Serializable]
    public struct FieldOverride
    {
        public string name;
        public AnimationCurve curve;
    }

    protected Dictionary<string, float> originalValues = new Dictionary<string, float>();

    protected TrainerParams parameters;
    protected Trainer trainer;

    private void Awake()
    {
        trainer = GetComponent<Trainer>();
        parameters = trainer.parameters;
    }

    private void FixedUpdate()
    {
        foreach(var o in overrides)
        {
            if (!originalValues.ContainsKey(o.name))
            {
                originalValues[o.name] = GetValue(o.name);
            }

            float value = o.curve.Evaluate(Mathf.Clamp01(((float)trainer.GetStep()) / trainer.GetMaxStep())) * originalValues[o.name];
            SetValue(o.name, value);
        }
    }

    private void SetValue(string name, float value)
    {
        var fieldInfo = parameters.GetType().GetField(name);
        if(fieldInfo != null)
        {
            fieldInfo.SetValue(parameters, value);
        }
    }

    private float GetValue(string name)
    {
        var fieldInfo = parameters.GetType().GetField(name);
        if (fieldInfo != null)
        {
            return (float)fieldInfo.GetValue(parameters);
        }
        return 0;
    }

    private void OnDisable()
    {
        foreach(var v in originalValues)
        {
            SetValue(v.Key, v.Value);
        }
    }
}
