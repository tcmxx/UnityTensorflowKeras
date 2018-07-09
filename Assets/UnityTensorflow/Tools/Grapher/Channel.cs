using System;
using System.Collections.Generic;
#if UNITY_EDITOR

using UnityEditor;
using UnityEngine;

namespace NWH
{
    public class Channel
    {
        public Grapher g;
        public List<Sample> rawSampleList;

        public int id;
        public string name;
        public Color color;
        public float tagY;
        public float tagX;

        private float timeScale = 5;
        private float yMax;
        private float yMin;
        public int sampleNo = 0;
        public bool beingManuallyAdjusted = false;
        public float autoScaleResolution = GraphSettings.DefaultVerticalResolution;

        public int lastFrame = 0;

        // Sliders
        public float timeScaleSlider;
        public float rangeSlider;

        // Marker
        public Vector2 pointAtMousePosition;

        public int lastVisiblePointIndex;
        public int firstVisiblePointIndex;

        public Sample newestSample;
        public object newestObj;

        // Scale
        public float xScale;
        public float yScale;

        public float xOffset;
        public float yOffset;

        // Replay
        public bool replay = false;
        public bool replayEnded = false;

        public void Init()
        {
            sampleNo = 0;
            rawSampleList = new List<Sample>();
        }

        public bool Show
        {
            get
            {
                string key = "Grapher" + name + "Show";
                return EditorPrefs.HasKey(key) ? EditorPrefs.GetBool(key, true) : true;
            }
            set
            {
                string key = "Grapher" + name + "Show";
                EditorPrefs.SetBool(key, value);
            }
        }

        public bool LogToFile
        {
            get
            {
                string key = "Grapher" + name + "LogToFile";
                return EditorPrefs.HasKey(key) ? EditorPrefs.GetBool(key, true) 
                    : GraphSettings.DefaultLogToFile == 1 ? true : false;
            }
            set
            {
                string key = "Grapher" + name + "LogToFile";
                EditorPrefs.SetBool(key, value);
            }
        }

        public bool LogToConsole
        {
            get
            {
                string key = "Grapher" + name + "LogToConsole";
                return EditorPrefs.HasKey(key) ? EditorPrefs.GetBool(key, true) 
                    : GraphSettings.DefaultLogToConsole == 1 ? true : false;
            }
            set
            {
                string key = "Grapher" + name + "LogToConsole";
                EditorPrefs.SetBool(key, value);
            }
        }

        public bool AutoScale
        {
            get
            {
                string key = "Grapher" + name + "AutoScale";
                return EditorPrefs.HasKey(key) ? EditorPrefs.GetBool(key, true) : true;
            }
            set
            {
                string key = "Grapher" + name + "AutoScale";
                EditorPrefs.SetBool(key, value);
            }
        }

        public float TimeScale
        {
            get
            {
                return timeScale;
            }
            set
            {
                timeScale = Mathf.Clamp(value, 0.5f, 3600f);
            }
        }

        public float verticalResolution
        {
            get
            {
                string key = "Grapher" + name + "verticalResolution";
                return EditorPrefs.HasKey(key) ? EditorPrefs.GetFloat(key) : GraphSettings.DefaultVerticalResolution;
            }
            set
            {
                string key = "Grapher" + name + "verticalResolution";
                float range = Mathf.Clamp(value, 0.00001f, 100000000f);
                EditorPrefs.SetFloat(key, range);
            }
        }

        public float YMin { get { return yMin; } }
        public float YMax { get { return yMax; } }

        public Channel(int id)
        {
            this.id = id;
        }

        public void Enqueue(float x, float t)
        {
            Sample sample = new Sample(x, t);

            if (rawSampleList == null) rawSampleList = new List<Sample>();
            rawSampleList.Add(sample);
            sampleNo++;

            // Determine max and min
            if (sampleNo <= 2f)
            {
                yMax = x;
                yMin = x;
            }
            else if (x > yMax)
            {
                yMax = x;
            }
            else if (x < yMin)
            {
                yMin = x;
            }

            // Get auto range
            autoScaleResolution = Mathf.Max(Mathf.Abs(yMin), Mathf.Abs(yMax)) * 2f;
        }

        public Sample[] GetSamples()
        {
            if (rawSampleList != null)
            {
                return rawSampleList.ToArray();
            }
            else
            {
                return null;
            }
        }

        public void ResetSamples()
        {
            rawSampleList.Clear();
            sampleNo = 0;
            timeScale = GraphSettings.HorizontalResolution;
            verticalResolution = GraphSettings.DefaultVerticalResolution;
        }
    }
}

#endif