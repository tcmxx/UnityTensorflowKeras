using System;

#if UNITY_EDITOR

using UnityEditor;

namespace NWH
{
    public static class TimeKeeper
    {
        public static double systemTime;
        private static double prevSystemTime;
        public static float systemDeltaTime;

        public static string DateTimeString { get { return DateTime.Now.ToString("yyyy-MM-ddTHH:mm:sszzz"); } }
        public static void Update()
        {
            systemTime = TimeSpan.FromTicks(DateTime.Now.Ticks).TotalSeconds;
            if (prevSystemTime == 0) prevSystemTime = systemTime;
            systemDeltaTime = (float)(systemTime - prevSystemTime);
            
            prevSystemTime = systemTime;
        }
        

    }
}
#endif