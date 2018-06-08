using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace NWH
{
    [System.Serializable]
    public class Sample
    {
        public float t;
        public float d;

        public Sample(float d, float t)
        {
            this.t = t;
            this.d = d;
        }
    }
}
