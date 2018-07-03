using System.Collections;
using System.Collections.Generic;
#if UNITY_EDITOR

using UnityEditor;
using UnityEngine;

namespace NWH
{
    [System.Serializable]
    public static class GraphSettings
    {
        // Positions
        public static Vector4 graphMargins = new Vector4(0f, 1f, 305f, 40f);

        // UI
        public static float sliderSensitivity = 0.5f;
        public static float chButtonSize = 18f;

        // Graph 
        public static bool showHorizontalRule = false;
        public static bool showVerticalRule = true;

        // Colors
        public static Color32 windowBackgroundColor = new Color32(162, 162, 162, 255);
        public static Color32 borderBackgroundColor = new Color32(194, 194, 194, 255);
        public static Color32 graphBackgroundColor = new Color32(80, 100, 100, 255);
        public static Color32 graphStatColor = new Color(0, 0, 0, 0.4f);
        public static Color32 ruleLineColor = new Color32(255, 255, 255, 100);
        public static Color32 gridLineColor = new Color32(255, 255, 255, 80);

        public static Color32 panelBackgroundColor = new Color32(240, 240, 240, 255);
        public static Color32 panelHeaderColor = new Color32(220, 220, 220, 255);
        public static Color32 buttonBackgroundColor = new Color32(220, 220, 220, 255);
        public static Color32 buttonHoverColor = new Color32(180, 180, 180, 255);
        public static Color32 buttonActiveColor = new Color32(130, 255, 150, 255);

        public static float autoScalePercent = 0.9f;

        public static float HorizontalResolution
        {
            get
            {
                return EditorPrefs.GetFloat("GrapherHorizontalResolution", 8);
            }
            set
            {
                EditorPrefs.SetFloat("GrapherHorizontalResolution", value);
            }
        }

        public static int SharedVerticalResolution
        {
            get
            {
                return EditorPrefs.GetInt("GrapherSharedVerticalResolution", 0);
            }
            set
            {
                EditorPrefs.SetInt("GrapherSharedVerticalResolution", value);
            }
        }

        public static int OverwriteFiles
        {
            get
            {
                return EditorPrefs.GetInt("GrapherOverwriteFiles", 0);
            }
            set
            {
                EditorPrefs.SetInt("GrapherOverwriteFiles", value);
            }
        }
        public static int SaveWhenStopped
        {
            get
            {
                return EditorPrefs.GetInt("SaveWhenStopped", 0);
            }
            set
            {
                EditorPrefs.SetInt("SaveWhenStopped", value);
            }
        }
        public static int GraphLineStyle
        {
            get
            {
                return EditorPrefs.GetInt("GrapherLineStyle", 0);
            }
            set
            {
                EditorPrefs.SetInt("GrapherLineStyle", value);
            }
        }

        public static float DefaultVerticalResolution
        {
            get
            {
                return EditorPrefs.GetFloat("GrapherDefaultVerticalResolution", 10);
            }
            set
            {
                EditorPrefs.SetFloat("GrapherDefaultVerticalResolution", value);
            }
        }

        public static int DefaultLogToFile
        {
            get
            {
                return EditorPrefs.GetInt("GrapherLogToFileDefault", 0);
            }
            set
            {
                EditorPrefs.SetInt("GrapherLogToFileDefault", value);
            }
        }

        public static int DefaultLogToConsole
        {
            get
            {
                return EditorPrefs.GetInt("GrapherLogToConsoleDefault", 0);
            }
            set
            {
                EditorPrefs.SetInt("GrapherLogToConsoleDefault", value);
            }
        }

        public static Texture2D GetTextureFrom64(string b64)
        {
            Texture2D texture = new Texture2D(1, 1);
            var b64Bytes = System.Convert.FromBase64String(b64);
            texture.LoadImage(b64Bytes);
            texture.Apply();
            return texture;
        }


        //Icons in b64 (a very hairy workaround for asset store guidelines problems)
        public static string scaleIcon64 = @"iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABHNCSVQICAgIfAhkiAAAAAl
                wSFlzAAAAwgAAAMIBT4kc1wAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAADvSURBVEiJtdUxTsNAEEbhT7Q0CBGRMuDchcumCTI
                SNU1aBEK0Jj5BxAGWxpaCsdfLbjLSVDt6b+fXWhZCkNuoESL9fKGsLmfOH7Jv321wjdfoFiWCTnKD9ylBaURw1fV4Fd5+jfYsEU3A2z9xnRC+R4UF3rI
                FEfj90cwCL6hzBI8D+BfuJuczBFUHDWhi8CxBJ1lhg9XsbCJsi3XWZRLgzdET/LckFd73NiNONXZYzsCblMzHBD3gE8sJePQppgp6yRC+z4WPCYb96ws
                9taBFVQIPIYj9Dw74jpwnVyyiD9yeK6K+d6URPc1seCiJ5wc6yz713wYsOQAAAABJRU5ErkJggg==";
        public static string logIcon64 = @"iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8 / 9hAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAEhQ
                AABIUBEapzDgAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAADaSURBVDiNrZK9DkFBEIW/8ZMoVCqNRCGIaHTeQqVReQSJhlbDQ0i0HkHiBXQaxY
                1eoZQoJUdhb7K4WOEkk83OzpyZM7MGRECNG7bAntfIAR1gJmkcOyNAzgaS8A1IA32gBxS92KkkMm+qxcgCC+AE1D3/yMwuqQCCd+iGdCDg6Dp4wiNBwczKCXFtd5aSSP
                whfmvRrzO4k3AA5gE5eWDoO2IJG7f3KjfNzcc/4d79vxAlbaHlgs7A7hsJMWpAA8DMGsBa0iqYQNLkU1UfP2/hr2usmNkyICfnX658BVigXx2yfwAAAABJRU5ErkJggg==";
        public static string consoleIcon64 = @"iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABHNCSVQICAgIfAhkiAAAA
                AlwSFlzAAAAeAAAAHgB6vJq9gAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAACmSURBVDiNpZNBCsJQDETffHHlTrvsFbyCC8/hNT
                yPtEfxAFLdegNxK4ggcdNQBNua/kAgWcwwzCQyM3IqZaGBJKmQNMtRcAQqSZPVbIEHUAPJzIg0rYlOUkVJuqEjOURIvhfYAQbspygogStwApZRDxzcRMB
                +hA4+A6toCgIuHqeZ3T1bSXNg7MDeABug+CHt1ho61C/1PZOkNbAYUfDsJfi3sr/xA0B1MttepcDyAAAAAElFTkSuQmCC";
        public static string showIcon64 = @"iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
                AAAFnAAABZwBCcboxgAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAHfSURBVEiJ7dW9S5dRFAfwzzVNNBEianFuKSpzqBZxaWlyaWlX
                QmjoRaTcxYgkHIRqa4qwudYGsRDSTAr6A6KhJLDNXk7DvT/8+fxeaHEJLxx4zrnP/Z5zvvecc1NE2MvVsafo+w7+ZXW220wpHcQpnMUQzuAP3mEVa/gQET9b
                YjSropTSEdzGJbwpYO+xLWfdjdPF6Tk8x1xEbDWARcQuwSi+YaYA9WGhOHlSZA0P0FtkHl8w0oBXAZ8uh4eK3oe3GCv6QN2/E1hBT9GH8RHjTR1gFo/QVWdb
                wBguYgn3KgFdK9TU9F4s4uYuBwX8ce1Oiq0Tq+V7CUeb0JnkC6+eW8RkROhIKc3iGK7G7hs/gY2U0gCWI+Jrk/sLfMLxOtsvXMH5lNJUB76jHwcaKiAf+BwR
                U8322qzf2MLhWlqTJa3OSqprVVqaULReoSjJdM9GRO7kiLiP13iaUuqtS/VVSmmiTaTX8aJGbUqpCw+xGRF3ahTURzQul9pwXVWsyNVSjfIGltFdbENyiU+37
                IPy44jcNPPFQQ/m5Gp5VmRdrrzuIjNyc45W8VqNin7cwuWSwWpx8AOBQxgsUV/AS9yNiM0GrHYvWuH0pJ1hNyjPonU7w24jIrZbYuw/mf+/g7/+5GhqtHt1wg
                AAAABJRU5ErkJggg==";
    }

}

#endif