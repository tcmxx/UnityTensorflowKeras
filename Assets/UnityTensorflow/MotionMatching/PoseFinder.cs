using MotionMatching;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
namespace MotionMatching
{
    public class PoseFinder : MonoBehaviour
    {


        public bool verbose = false;
        public Transform movingRoot;
        /// <summary>
        /// Generated pose data.
        /// </summary>
        public MotionPose[][] Poses { get; private set; }

        /// <summary>
        /// Current animation index, refers to clip index.
        /// </summary>
        private int currentAnimIndex;
        /// <summary>
        /// Current animation playback time in seconds.
        /// </summary>
        private float currentAnimTime;
        /// <summary>
        /// Currently played animation length in seconds.
        /// </summary>
        private float currentAnimLength;

        [Tooltip("Animation clips used in the motion matching.")]
        public AnimationClip[] clips;

        [Tooltip("Time in seconds between generated poses.")]
        public float poseInterval = 0.1f;


        [Tooltip("Extra bones to be used in the pose matching.")]
        public Transform[] poseMatchingBones = new Transform[0];

        public float rootAngularVelocityWeight = 1;
        public float rootVelocityWeight = 1;
        public float jointAngularVelocityWeight = 1;
        public float jointRotationWeight = 1;
        public float jointPositionWeight = 1;

        private void Awake()
        {
        }

        private void Start()
        {
            Vector3 initialPosition = movingRoot.position;
            Quaternion initialRotation = movingRoot.rotation;

            // Generate pose data.
            Poses = PoseGenerator.GeneratePoseData(clips, movingRoot, gameObject, poseMatchingBones, poseInterval);

            currentAnimIndex = 0;
            currentAnimTime = 0;
            currentAnimLength = clips[0].length;

            //clips[1].SampleAnimation(gameObject, 0.5f);

        }






        public MotionPose FindBestPoseNext(MotionPose fromPose)
        {
            var best = FindBestPose(fromPose);
            return Poses[best.animIndex][(best.index + 1) % Poses[best.animIndex].Length];
        }

        public MotionPose FindBestPose(MotionPose fromPose)
        {
            // Determine the next best pose to jump to.
            float bestCost = float.MaxValue;
            int bestAnim = 0, bestPose = 0;

            // Cache transform values.
            Vector3 position = transform.position;
            float rotation = transform.rotation.eulerAngles.y;

            for (int i = 0; i < Poses.Length; i++)
            {
                for (int j = 0; j < Poses[i].Length; j++)
                {
                    MotionPose candidate = Poses[i][j];

                    // Calculate jumping cost
                    float cost = ComputeCurrentCost(fromPose, candidate);

                    if (cost < bestCost)
                    {
                        bestAnim = i;
                        bestPose = j;
                        bestCost = cost;
                    }
                    //Debug.LogFormat("animIndex {0}, pose {1}, cost {2}", i, j, cost);
                }
            }


            if (verbose)
                Debug.LogFormat("Jumping: BestCost {0}, animIndex {1}, pose{2}, animTime {3}, prevAnimIndex: {4}, prevAnimTime: {5}",
                    bestCost, bestAnim, bestPose, Poses[bestAnim][bestPose].animTime, currentAnimIndex, currentAnimTime);
            return Poses[bestAnim][bestPose];


        }

        public void GotoPose(MotionPose pose)
        {
            for (int i = 0; i < poseMatchingBones.Length; ++i)
            {
                poseMatchingBones[i].position = movingRoot.TransformPoint(pose.jointPositions[i]);
                poseMatchingBones[i].localRotation = pose.jointLocalRotations[i];
            }

        }




        /// <summary>
        /// Computes cost between two poses.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        private float ComputeCurrentCost(MotionPose a, MotionPose b)
        {
            float cost = 0f;

            // Root velocity cost
            cost += (a.velocity - b.velocity).magnitude * rootVelocityWeight;
            // Difference between angular velocity.
            float angle = (a.angularVelocity - b.angularVelocity).magnitude * rootAngularVelocityWeight;
            cost += NormalizeAngle(angle);

            // Joint position cost
            for (int i = 0; i < a.jointPositions.Length; i++)
                cost += (a.jointPositions[i] - b.jointPositions[i]).magnitude * jointPositionWeight;

            // Joint velocity cost
            for (int i = 0; i < a.jointLocalRotations.Length; i++)
                cost += MotionMatchingUtils.NormalizeAngle((a.jointLocalRotations[i]*Quaternion.Inverse(b.jointLocalRotations[i])).eulerAngles).magnitude*jointRotationWeight;

            // Joint angular velocity cost
            for (int i = 0; i < a.jointAngularVelocity.Length; i++)
                cost += (a.jointAngularVelocity[i] - b.jointAngularVelocity[i]).magnitude * jointAngularVelocityWeight;



            return cost;
        }

        /// <summary>
        /// Computes delta position by summing the delta velocity
        /// from all poses which are wihtin the given duration
        /// from the start pose.
        /// 
        /// NOTE! Assumes that all animations are loopable!
        /// </summary>
        /// <param name="animIndex"></param>
        /// <param name="startPoseIndex"></param>
        /// <param name="duration"></param>
        /// <returns></returns>
        private Vector3 ComputeVelocity(int animIndex, int startPoseIndex, float duration)
        {
            Vector3 result = Vector3.zero;
            float elapsed = 0f;

            int i = startPoseIndex;
            do
            {
                // Add velocity
                result += Poses[animIndex][i].velocity;

                // Wrap around
                i = i + 1 >= Poses[animIndex].Length ? 0 : i + 1;

                elapsed += poseInterval;

            } while (elapsed < duration);

            return result;
        }

        /// <summary>
        /// Normalizes given angle to range [0, 1]
        /// </summary>
        /// <param name="angle"></param>
        /// <returns></returns>
        private float NormalizeAngle(float angle)
        {
            return Mathf.Abs(angle) / 180f;
        }
    }
}