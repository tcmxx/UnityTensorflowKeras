using UnityEngine;

namespace MotionMatching
{
    /// <summary>
    /// Struct for motion matching goals.
    /// </summary>
    public struct Goal
    {
        public TrajectoryPoint[] desiredTrajectory;
    }

    /// <summary>
    /// Struct for trajectory points.
    /// </summary>
    public struct TrajectoryPoint
    {
        /// <summary>
        /// Trajectory point in world space.
        /// </summary>
        public Vector3 position;

        /// <summary>
        /// Y -axis rotation in world space.
        /// </summary>
        public float orientation;

        /// <summary>
        /// Time delay in seconds.
        /// </summary>
        public float timeDelay;
    }
}