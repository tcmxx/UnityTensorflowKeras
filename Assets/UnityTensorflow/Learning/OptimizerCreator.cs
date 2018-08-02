using KerasSharp.Optimizers;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[Serializable]
public class OptimizerCreator  {
    public enum OptimizerType
    {
        Adam,
        SGD,
        RMSProp
    }

    public OptimizerType optimizerType;
    [SerializeField]
    public List<float> parameterList = new List<float>();
    
    public OptimizerBase CreateOptimizer()
    {
        Type type = TypeFromEnum(optimizerType);
        var ctors = type.GetConstructors();
        var ctor = ctors[0];    //assume there is only one constructor
        var paramInfos = ctor.GetParameters();
        List<object> parameters = new List<object>();

        var paraminfors = ctor.GetParameters();
        int i = 0;
        foreach (var param in paraminfors)
        {
            if (parameterList.Count > i && (param.ParameterType == typeof(float) || 
                param.ParameterType == typeof(double) || 
                param.ParameterType == typeof(int) || 
                param.ParameterType == typeof(bool)))
            {
                if(param.ParameterType == typeof(float))
                {
                    parameters.Add(parameterList[i]);
                }
                else if (param.ParameterType == typeof(double))
                {
                    //Debug.Log(i +  "/" + parameterList.Count);
                    parameters.Add((double)parameterList[i]);
                }
                else if (param.ParameterType == typeof(int))
                {
                    parameters.Add((int)parameterList[i]);
                }
                else if (param.ParameterType == typeof(bool))
                {
                    parameters.Add(parameterList[i]>0);
                }
                
            }
            else if(param.HasDefaultValue)
            {
                parameters.Add(param.RawDefaultValue);
            }
            else
            {
                parameters.Add(0);
            }

            i++;
        }

        return (OptimizerBase)ctor.Invoke(parameters.ToArray());
    }

    public static Type TypeFromEnum(OptimizerType type)
    {
        switch (type)
        {
            case OptimizerType.Adam:
                return typeof(Adam);
            case OptimizerType.RMSProp:
                return typeof(RMSProp);
            case OptimizerType.SGD:
                return typeof(SGD);
            default:
                throw new NotImplementedException();
        }
    }


}
