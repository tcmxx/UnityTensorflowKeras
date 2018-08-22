using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using UnityEngine;

public class StatsLogger
{
    protected Dictionary<string, List<ValueTuple<float, float>>> loggedData;

    protected Dictionary<string, List< float>> tempData;

    public bool LogToGrapher { get; set; } = true;

    protected MethodInfo logMethodInfo = null;
    protected MethodInfo resetMethodInfo = null;

    public StatsLogger()
    {
        loggedData = new Dictionary<string, List<ValueTuple<float, float>>>();
        tempData = new Dictionary<string, List<float>>();

        Type type = Type.GetType("Grapher");
        if (type != null)
        {
            logMethodInfo = type.GetMethod("Log", new Type[] { typeof(float), typeof(string), typeof(float) });
            resetMethodInfo = type.GetMethod("Reset", new Type[] {});
        }
    }


    public void LogAllCurrentData(float currentStep)
    {

        foreach(var k in loggedData.Keys)
        {
            LogCurrentData(k, currentStep);
        }
    }

    public void LogCurrentData(string name, float currentStep)
    {

        if(tempData.ContainsKey(name) && tempData[name].Count > 0)
        {
            float ave = 0;
            int count = tempData[name].Count;
            for (int i = 0; i < count; ++i) {
                ave += tempData[name][i];
            }
            
            ave = ave / count;
            tempData[name].Clear();

            loggedData[name].Add(ValueTuple.Create(currentStep, ave));

#if UNITY_EDITOR
            //Grapher.Log(averageCounter[name].Average, name);
            if (logMethodInfo != null)
            {
                object[] parametersArray = new object[] { ave, name, currentStep };
                logMethodInfo.Invoke(null, parametersArray);

            }
#endif
        }


    }



    public void AddData(string name, float datapoint)
    {
        if (!loggedData.ContainsKey(name))
        {
            loggedData[name] =  new List<ValueTuple<float, float>>();
            tempData[name] =  new List<float>();
        }
        
        tempData[name].Add(datapoint);
    }


    public List<ValueTuple<float,float>> GetStat(string name)
    {
        return loggedData.TryGetOr(name, null);
    }

    public void Clear(string name)
    {
        tempData.Remove(name);
        loggedData.Remove(name);
    }
    public void ClearAll()
    {
        tempData.Clear();
        loggedData.Clear();
#if UNITY_EDITOR
        //Grapher.Log(averageCounter[name].Average, name);
        if (resetMethodInfo != null)
        {
            resetMethodInfo.Invoke(null, null);

        }
#endif
    }
}
