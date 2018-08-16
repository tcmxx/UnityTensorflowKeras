using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(Trainer))]
public class TrainerParamOverride : MonoBehaviour {

    

    public FieldOverride[] overrides;

    [Serializable]
    public class FieldOverride
    {
        public string name;
        public Method method;
        public AnimationCurve curve;
        public float endValue;
        public float power;
    }

    public enum Method
    {
        AnimationCurve,
        PolynomialDecay
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

            if (o.method == Method.AnimationCurve)
            {
                float value = o.curve.Evaluate(Mathf.Clamp01(((float)trainer.GetStep()) / trainer.GetMaxStep())) * originalValues[o.name];
                SetValue(o.name, value);
            }else if(o.method == Method.PolynomialDecay)
            {
                float value = (originalValues[o.name] - o.endValue)*Mathf.Pow(1-((float)trainer.GetStep())/trainer.GetMaxStep(), o.power) +o.endValue;
            }
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
