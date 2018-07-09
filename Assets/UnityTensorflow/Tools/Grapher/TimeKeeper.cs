using System;

#if UNITY_EDITOR

using UnityEditor;

namespace NWH
{
    public static class TimeKeeper
    {
        private static float time = 0;

        public static double systemTime;
        private static double prevSystemTime;
        public static float systemDeltaTime;

        public static float Time
        {
            get
            {
                return time;
            }
            set
            {
                time = value;
                EditorPrefs.SetFloat("GrapherTime", time);
            }
        }

        public static void Update()
        {
            systemTime = TimeSpan.FromTicks(DateTime.Now.Ticks).TotalSeconds;
            if (prevSystemTime == 0) prevSystemTime = systemTime;
            systemDeltaTime = (float)(systemTime - prevSystemTime);

            // Check if time should run
            if ((EditorApplication.isPaused || Grapher.replayControl == Grapher.ReplayControls.Pause)
                || (Grapher.replayControl == Grapher.ReplayControls.Stop && (!EditorApplication.isPaused && !EditorApplication.isPlaying)))
            {
                time = EditorPrefs.GetFloat("GrapherTime", 0);
            }
            else
            {
                time += systemDeltaTime;
                EditorPrefs.SetFloat("GrapherTime", Time);
            }

            prevSystemTime = systemTime;
        }

        public static void Reset()
        {
            time = 0;
            EditorPrefs.SetFloat("GrapherTime", 0);
        }

    }
}
#endif