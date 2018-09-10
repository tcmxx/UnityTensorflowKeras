using UnityEngine;

#if UNITY_EDITOR
using UnityEditor.Animations;
#endif

namespace MotionMatching
{
    public static class MotionMatchingUtils
    {

        /// <summary>
        /// Creates multi state animation controlled with given clips.
        /// </summary>
        /// <param name="animator"></param>
        /// <param name="clips"></param>
        /// <param name="stateNamePrefix"></param>
        public static void SetMultiStateAnimation(Animator animator, AnimationClip[] clips, string stateNamePrefix = "")
        {
#if UNITY_EDITOR
            AnimatorController c = AnimatorController.CreateAnimatorControllerAtPathWithClip("Assets/TestController.controller", clips[0]);

            AnimatorStateMachine sm = c.layers[0].stateMachine;
            for (int i = 0; i < clips.Length; i++)
            {
                // Animation clip
                AnimatorState state = sm.AddState(stateNamePrefix + i);
                state.motion = clips[i];

                // Transition trigger
                Debug.Log("animatorState.name " + state.name);
                c.AddParameter(state.name, AnimatorControllerParameterType.Trigger);

                // Transition
                AnimatorStateTransition transition = sm.AddAnyStateTransition(state);
                transition.hasExitTime = false;
                transition.canTransitionToSelf = true;

                transition.AddCondition(AnimatorConditionMode.If, 1, state.name);
            }

            AnimatorController.SetAnimatorController(animator, c);
#else
            Debug.LogError("Does not work outside Unity Editor!");
#endif
        }

        public static Goal InitializeGoal(int count)
        {
            return new Goal() { desiredTrajectory = new TrajectoryPoint[count] };
        }

        public static Vector3 NormalizeAngle(Vector3 angles)
        {
            angles.x = Mathf.Repeat(angles.x + 180, 360) - 180;
            angles.y = Mathf.Repeat(angles.y + 180, 360) - 180;
            angles.z = Mathf.Repeat(angles.z + 180, 360) - 180;
            return angles;
        }
    }
}