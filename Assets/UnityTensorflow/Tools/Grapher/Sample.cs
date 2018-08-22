using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace NWH
{
    [System.Serializable]
    public class Sample
    {
        public string time;
        public float y;
        public float x;
        public Sample(float y, string time, float x)
        {
            this.time = time;
            this.y = y;
            this.x = x;
        }
    }
}
