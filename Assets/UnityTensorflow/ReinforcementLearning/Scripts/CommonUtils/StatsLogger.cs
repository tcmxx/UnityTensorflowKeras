using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class StatsLogger
{
    protected Dictionary<string, List<float>> data;

    protected Dictionary<string, AutoAverage> averageCounter;

    public bool LogToGrapher { get; set; } = true;

    public StatsLogger()
    {
        data = new Dictionary<string, List<float>>();
        averageCounter = new Dictionary<string, AutoAverage>();
    }



    public void AddData(string name, float datapoint, int logAverageFrequency = 1)
    {
        if (!data.ContainsKey(name))
        {
            data[name] = new List<float>();
            averageCounter[name] = new AutoAverage(logAverageFrequency);
        }

        data[name].Add(datapoint);
        averageCounter[name].AddValue(datapoint);
        if (LogToGrapher && averageCounter[name].JustUpdated)
        {
            Grapher.Log(averageCounter[name].Average, name);
        }
    }

    public List<float> GetStat(string name)
    {
        return data.TryGetOr(name, null);
    }

    public void Clear(string name)
    {
        data.Remove(name);
        averageCounter.Remove(name);
    }
    public void ClearAll()
    {
        data.Clear();
        averageCounter.Clear();
    }
}
