using UnityEngine;
using System.Linq;

namespace MotionMatching
{
    /// <summary>
    /// Static class which generates Pose data.
    /// </summary>
    public static class PoseGenerator {

        /// <summary>
        /// Static function which returns generated Poses using given Animator and AnimationClips.
        /// </summary>
        /// <param name="animator">Animator component of the character.</param>
        /// <param name="clips">Animation clips of the animation from which Poses are generated.</param>
        /// <param name="root">Root Transform of the character. Used to calculate joint positions.</param>
        /// <param name="jointTransforms">All extra joint position and velocities to be added in the Pose data.</param>
        /// <param name="poseInterval">Time in seconds between poses.</param>
        /// <returns></returns>
        public static MotionPose[][] GeneratePoseData(AnimationClip[] clips, Transform movingRoot, GameObject animatorRoot, Transform[] jointTransforms, float poseInterval)
        {
            // Initialize Pose storage.
            MotionPose[][] poses = new MotionPose[clips.Length][];

            // Root position in the previous Pose
            Vector3 prevPosePosition;

            // Rotation in the previous Pose.
            Quaternion prevRot;
            //animator.updateMode = AnimatorUpdateMode.AnimatePhysics;
            //Physics.autoSimulation = false;
            for (int i = 0; i < clips.Length; i++)
            {
                var clip = clips[i];

                // Change animation state clip state.
                // Calculate pose count.
                int poseCount = Mathf.RoundToInt(clip.length / poseInterval);
                Debug.LogFormat("Processing clip {0}, duration {1}, poseCount {2}", clip.name, clip.length, poseCount);

                // Store the initial position. 
                prevPosePosition = movingRoot.position;

                // Store the initial Y rotation.
                prevRot = movingRoot.rotation;

                // Initialize storage for framecount with store and 
                // store for first pose, which will be removed in the end.
                poses[i] = new MotionPose[poseCount + 1];
                
                // Loop through the animation and generate Pose data.
                for (int frame = 0; frame <= poseCount; frame++)
                {
                    // Initialize Pose storage.
                    var pose = new MotionPose()
                    {
                        index = frame - 1, // Correction for the pose index, because the first pose is removed.
                        animIndex = i,
                        animTime = frame * poseInterval,
                        isLooping = clip.isLooping,
                        isFromAnimation = true
                        
                    };

                    // Update animation.
                    //Physics.Simulate(poseInterval);
                    //animator.Update(poseInterval);
                    clip.SampleAnimation(animatorRoot.gameObject, pose.animTime);


                    // Pose velocity.
                    pose.velocity = (movingRoot.position - prevPosePosition)/ poseInterval;

                    // Angular velocity.
                    var deltaRot = movingRoot.rotation * Quaternion.Inverse(prevRot);

                    Vector3 normalizedDelta = MotionMatchingUtils.NormalizeAngle(deltaRot.eulerAngles);
                    pose.angularVelocity = normalizedDelta / poseInterval;
                    
                    // There is no velocity for the first pose 
                    // as it is used only as a reference for the next poses.
                    
                    pose.jointPositions = GetJointPosition(movingRoot, jointTransforms);
                    pose.jointLocalRotations = GetJointRot(jointTransforms);
                    if (frame > 0)
                        pose.jointAngularVelocity = GetJointAngularVelocities(pose.jointLocalRotations, poses[i][frame - 1].jointLocalRotations, poseInterval);


                    // Update prevPoseRootPosition.
                    prevPosePosition = movingRoot.position;

                    // Update prevRot
                    prevRot = movingRoot.rotation;
                    // Add pose to list.
                    poses[i][frame] = pose;
                }

                // Remove first element from the pose data.
                poses[i] = poses[i].Skip(1).ToArray();
            }

            clips[0].SampleAnimation(animatorRoot.gameObject, 0);
            return poses;
        }

        /// <summary>
        /// Returns array of position from jointTransforms.
        /// </summary>
        /// <param name="root"></param>
        /// <param name="jointTransforms"></param>
        /// <returns></returns>
        public static Vector3[] GetJointPosition(Transform root, Transform[] jointTransforms)
        {
            Vector3[] positions = new Vector3[jointTransforms.Length];
            // Transform Joint positions from local to global relative to root Transform.
            for (int i = 0; i < jointTransforms.Length; i++)
            {
                positions[i] = root.InverseTransformPoint(jointTransforms[i].position);
            }
            
            return positions;
        }

        public static Quaternion[] GetJointRot(Transform[] jointTransforms)
        {
            Quaternion[] rotations = new Quaternion[jointTransforms.Length];
            // Transform Joint positions from local to global relative to root Transform.
            for (int i = 0; i < jointTransforms.Length; i++)
            {
                rotations[i] = jointTransforms[i].localRotation;
            }

            return rotations;
        }


        /// <summary>
        /// Returns array of angular velocities between new positions and previous positions.
        /// </summary>
        /// <param name="jointTranforms"></param>
        /// <param name="previousPositions"></param>
        /// <returns></returns>
        private static Vector3[] GetJointAngularVelocities(Quaternion[] newRotations, Quaternion[] previousRotations, float deltaTima)
        {
            Vector3[] velocities = new Vector3[newRotations.Length];

            for (int i = 0; i < newRotations.Length; i++)
                velocities[i] = MotionMatchingUtils.NormalizeAngle(newRotations[i] *Quaternion.Inverse(previousRotations[i]).eulerAngles)/ deltaTima;

            return velocities;
        }


    }
}