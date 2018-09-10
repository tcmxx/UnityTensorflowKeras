using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using MotionMatching;
using System;

public class AnimationMatchinTest : AgentDependentDecision
{

    public GameObject exampleObject;
    protected PoseFinder poseFinder;

    protected Dictionary<Transform, Transform> parents;
    protected Dictionary<Transform, Rigidbody> jointsRB;
    protected Dictionary<Transform, Joint> joints;
    protected Dictionary<Transform, Vector3> anchorPoints;

    protected Rigidbody rootRigidBody = null;
    
    protected MotionPose savedPose;
    public bool savePose = false;
    public bool applyPose = false;
    public bool testButton = false;

    protected Transform[] poseMatchingBones;
    protected Transform movingRoot;

    protected bool initialized = false;
    // Use this for initialization
    void Start()
    {
        poseFinder = exampleObject.GetComponent<PoseFinder>();
        Debug.Assert(poseFinder != null, "Example needs a PoseFinder.cs attached.");
        //Initialize();
    }

    // Update is called once per frame
    void Update()
    {
        if (savePose)
        {
            savePose = false;
            savedPose = GetCurrentPose();
        }
        if (applyPose)
        {
            applyPose = false;
            poseFinder.GotoPose(savedPose);
        }
        if (testButton == true)
        {
            testButton = false;
            Test();
        }
    }

    protected void Initialize()
    {
        poseMatchingBones = new Transform[poseFinder.poseMatchingBones.Length];
        parents = new Dictionary<Transform, Transform>();
        jointsRB = new Dictionary<Transform, Rigidbody>();
        anchorPoints = new Dictionary<Transform, Vector3>();
        joints = new Dictionary<Transform, Joint>();

        for (int i = 0; i < poseFinder.poseMatchingBones.Length;++i)
        {
            var myBone = FindDeepChild( transform, poseFinder.poseMatchingBones[i].name);
            Debug.Assert(myBone != null, "Could not find the transform with name:" + poseFinder.poseMatchingBones[i].name + " in this object");
            parents[myBone] = FindDeepChild(transform, poseFinder.poseMatchingBones[i].parent.name);
            Debug.Assert(parents[myBone] != null, "Could not find the transform with name:" + poseFinder.poseMatchingBones[i].parent.name + " in this object");

            jointsRB[myBone] = myBone.GetComponent<Rigidbody>();
            joints[myBone] = myBone.GetComponent<Joint>();
            poseMatchingBones[i] = myBone;
            anchorPoints[myBone] = joints[myBone].anchor;
        }

        movingRoot = FindDeepChild(transform, poseFinder.movingRoot.name);
        Debug.Assert(movingRoot != null, "Could not find the transform with name:" + poseFinder.movingRoot.name + " in this object");
        rootRigidBody = movingRoot.GetComponent<Rigidbody>();
        initialized = true;
    }


    public MotionPose GetCurrentPose()
    {
        // Initialize Pose storage.
        var pose = new MotionPose()
        {
            isFromAnimation = false
        };

        // There is no velocity for the first pose 
        // as it is used only as a reference for the next poses.
        pose.jointPositions = GetJointPosition(movingRoot, poseMatchingBones);
        pose.jointLocalRotations = GetRelatedRotationToParent(poseMatchingBones);
        pose.jointAngularVelocity = GetJointAngularVelocity(poseMatchingBones);
        pose.velocity = rootRigidBody == null?Vector3.zero: poseFinder.movingRoot.InverseTransformVector(rootRigidBody.velocity);
        pose.angularVelocity = rootRigidBody.angularVelocity ;
        return pose;
    }

    protected Quaternion[] GetRelatedRotationToParent(Transform[] transforms)
    {
        Quaternion[] result = new Quaternion[transforms.Length];
        for (var j = 0; j < transforms.Length; ++j)
        {
            result[j] = Quaternion.Inverse(parents[transforms[j]].rotation) * transforms[j].rotation;
        }
        return result;
    }
    protected Vector3[] GetJointPosition(Transform root, Transform[] jointTransforms)
    {
        Vector3[] positions = new Vector3[jointTransforms.Length];
        // Transform Joint positions from local to global relative to root Transform.
        for (int i = 0; i < jointTransforms.Length; i++)
        {
            positions[i] = root.InverseTransformPoint(jointTransforms[i].TransformPoint(anchorPoints[jointTransforms[i]]));
        }

        return positions;
    }
    protected Vector3[] GetJointAngularVelocity(Transform[] jointTransforms)
    {
        Vector3[] result = new Vector3[jointTransforms.Length];
        for (var j = 0; j < jointTransforms.Length; ++j)
        {
            var rb = jointsRB[jointTransforms[j]];
            result[j] = rb.angularVelocity;
        }
        return result;
    }

    //Breadth-first search
    public static Transform FindDeepChild(Transform aParent, string aName)
    {
        var result = aParent.Find(aName);
        if (result != null)
            return result;
        foreach (Transform child in aParent)
        {
            result = FindDeepChild(child, aName);
            if (result != null)
                return result;
        }
        return null;

    }







    public void GetActinTowardNextPose()
    {
        var currentPose = GetCurrentPose();
        var diff = PoseAngularDiff(currentPose, poseFinder.FindBestPoseNext(currentPose));

        var mujucoAgent = GetComponent<MujocoUnity.MujocoAgent>();
        var actions = new float[diff.Length];
        for(int i = 0; i < actions.Length; ++i)
        {
            actions[i] = diff[i].z;
        }
        mujucoAgent.AgentAction(actions,null);
    }

    protected static Quaternion[] PoseAngularDiff(MotionPose fromPose, MotionPose toPose)
    {
        int length = fromPose.jointLocalRotations.Length;
        var result = new Quaternion[length];
        for (int i = 0; i < length; ++i)
        {
            result[i] =toPose.jointLocalRotations[i] *  Quaternion.Inverse(fromPose.jointLocalRotations[i]);
        }
        return result;
    }



    protected void Test()
    {
        poseFinder.GotoPose(poseFinder.FindBestPoseNext(GetCurrentPose()));
        GetActinTowardNextPose();
    }

    public override float[] Decide(List<float> vectorObs, List<Texture2D> visualObs, List<float> heuristicAction, List<float> heuristicVariance = null)
    {
        if (!initialized)
        {
            try
            {
                Initialize();
            }catch(Exception e)
            {
                Debug.LogWarning("Initlaize decision failed");
                return heuristicAction.ToArray();
            }
        }
        var currentPose = GetCurrentPose();
        var diff = PoseAngularDiff(currentPose, poseFinder.FindBestPoseNext(currentPose));

        var mujucoAgent = GetComponent<MujocoUnity.MujocoAgent>();
        var actions = new float[diff.Length];
        for (int i = 0; i < actions.Length; ++i)
        {
            float angle = diff[i].eulerAngles.z;
            actions[i] = angle > 180? angle-360:angle;
            actions[i] = -actions[i]/10;
        }
        return actions;
    }
}