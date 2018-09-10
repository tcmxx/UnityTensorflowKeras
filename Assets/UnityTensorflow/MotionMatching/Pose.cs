using UnityEngine;

namespace MotionMatching
{
    
    /// <summary>
    /// "Pose contains everything I need to compute the cost... 
    /// Just a vector of things"
    /// 
    /// -Simon Clavet
    /// </summary>
    public struct MotionPose
    {
        public bool isFromAnimation;
        /// <summary>
        /// Index of this Pose in the Pose list.
        /// Used as optimization to reduce lookup counts.
        /// </summary>
        public int index;

        /// <summary>
        /// Index to the original Animation Clip from
        /// which this Pose was generated.
        /// </summary>
        public int animIndex;
        
        /// <summary>
        /// Time in seconds from the Animation Clip
        /// where this Pose was generated.
        /// </summary>
        public float animTime;

        /// <summary>
        /// Root velocity of this pose.
        /// </summary>
        public Vector3 velocity;

        /// <summary>
        /// Angular velocity of this pose.
        /// </summary>
        public Vector3 angularVelocity;

        /// <summary>
        /// Pose's joint positions, relative to root.
        /// </summary>
        public Vector3[] jointPositions;

        /// <summary>
        /// Pose's joint velocities.
        /// </summary>
        public Vector3[] jointAngularVelocity;

        public Quaternion[] jointLocalRotations;

        /// <summary>
        /// Is the source animation set as looped animation.
        /// </summary>
        public bool isLooping;

        public override string ToString()
        {
            return "";
        }
    }
}