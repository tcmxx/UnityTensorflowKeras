using AaltoGames;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(PhysicsStorageBehaviour))]
public class BilliardArena : MonoBehaviour
{

    [ReadOnly]
    public float score = 0;
    public bool rewardShaping = true;
    public float physicsDrag = 0.5f;

    protected PhysicsStorageBehaviour physicsStorageBehaviour;

    protected GameObject whiteBall;

    protected Dictionary<GameObject, bool> ballsToPocket;
    protected Vector3[] pocketPositions;

    public float PredictedShotScore { get; protected set; } = 0;

    //some temp vars about save/load,evaluation related.
    protected float savedScore;
    protected Dictionary<GameObject, bool> saveBallsToPocket;
    protected Rigidbody[] simulationBodies;
    protected Vector3[] simulationPositions;
    protected Color drawColor;

    private void Awake()
    {
        InitializeArena();
    }

    public void InitializeArena()
    {
        physicsStorageBehaviour = GetComponent<PhysicsStorageBehaviour>();
        whiteBall = transform.Find("WhiteBall").gameObject;

        ballsToPocket = new Dictionary<GameObject, bool>();
        Rigidbody[] bodies = GetComponentsInChildren<Rigidbody>();
        foreach (Rigidbody b in bodies)
        {
            if (b.gameObject != whiteBall)
                ballsToPocket[b.gameObject] = true;
            b.angularDrag = physicsDrag;
            b.drag = physicsDrag;
        }
        BilliardPocket[] pockets = GetComponentsInChildren<BilliardPocket>();
        pocketPositions = new Vector3[pockets.Length];
        for (int i = 0; i < pockets.Length; i++)
        {
            pocketPositions[i] = pockets[i].transform.position;
            pockets[i].arena = this;
        }


        BilliardBoundary[] bound = GetComponentsInChildren<BilliardBoundary>();
        for (int i = 0; i < bound.Length; i++)
        {
            bound[i].arena = this;
        }
    }


    // Update is called once per frame
    void Update()
    {
        //GraphUtils.DrawPendingLines();
    }

    public void Shoot(Vector3 force)
    {
        Rigidbody r = whiteBall.GetComponent<Rigidbody>();
        r.velocity = force;
        //Uncomment the following to also set the angular velocity of the ball such that it agrees with velocity.
        //This should prevent the ball from first sliding and then decelerating rapidly. However, it doesn't seem to work in practice,
        //and better results are obtained simply by having everything with zero friction and basically no rotation (which however looks ok only without textures...)
        //Vector3 axis = Vector3.Cross(Vector3.up, force.normalized).normalized; //this is the axis of rotation
        //float ballRadius = whiteBall.transform.localScale.x * 0.5f;
        //r.angularVelocity = 10.0f*axis * force.magnitude / (ballRadius * 2.0f * Mathf.PI);
    }
    public void OnPocket(GameObject ball)
    {
        if (ball == whiteBall)
        {
            score -= 10;
        }
        else
        {
            score += 1;
            if (!ballsToPocket.ContainsKey(ball))
            {
                var keys = new List<GameObject>(ballsToPocket.Keys);
                Debug.LogError("Other ball into the pocket. Ball of arena " + ball.transform.parent.name + ", arena is " + name + " own ball is " + keys[0].transform.parent.name);

            }
            ballsToPocket[ball] = false;
        }
        ball.SetActive(false);
    }

    public void OnoutOfBound(GameObject ball)
    {

        score -= 10;
        if (!ballsToPocket.ContainsKey(ball) && ball != whiteBall)
        {
            Debug.LogError("Other ball into the pocket");
        }
        
        ball.SetActive(false);
    }


    public bool ShotComplete()
    {
        return physicsStorageBehaviour.IsAllSleeping();
        //check whether any physics object is moving
        /*Rigidbody[] bodies = GameObject.FindObjectsOfType<Rigidbody>();
        foreach (Rigidbody b in bodies)
        {
            if (b.velocity.magnitude > 0.001f)
                return false;
        }
        return true;*/
    }
    public void StopAll()
    {
        physicsStorageBehaviour.StopAll();
    }

    public void SaveState()
    {
        physicsStorageBehaviour.SaveState();
        savedScore = score;
        saveBallsToPocket = new Dictionary<GameObject, bool>(ballsToPocket);
    }
    public void RestoreState()
    {
        physicsStorageBehaviour.RestoreState();
        score = savedScore;
        ballsToPocket = saveBallsToPocket;
    }



    public void StartEvaluateShot(Vector3 force, Color drawColor)
    {

        SaveState();

        //initialize score to 0, want to count only this shot
        score = 0;


        //initialize shot
        Shoot(force);

        this.drawColor = drawColor;

        //some helpers
        simulationBodies = GetComponentsInChildren<Rigidbody>();
        simulationPositions = new Vector3[simulationBodies.Length];

    }

    public void BeforeEvaluationUpdate()
    {
        for (int i = 0; i < simulationPositions.Length; i++)
            simulationPositions[i] = simulationBodies[i].position;
    }
    public void AfterEvaluationUpdate()
    {
        for (int i = 0; i < simulationPositions.Length; i++)
            if (simulationBodies[i].gameObject.activeSelf)
                //Debug.DrawLine(pos[i], bodies[i].position,Color.green);
                GraphUtils.AddLine(simulationPositions[i], simulationBodies[i].position, drawColor);
    }

    public float EndEvaluationShoot()
    {
        if (rewardShaping)
        {
            //Since the score as such provides very little gradient, we add a small score if the balls get close to the pockets
            foreach (var b in ballsToPocket)
            {
                if (b.Value)
                {
                    Vector3 ballPos = b.Key.transform.position;
                    float minSqDist = float.MaxValue;
                    for (int i = 0; i < pocketPositions.Length; i++)
                    {
                        minSqDist = Mathf.Min(minSqDist, (pocketPositions[i] - ballPos).sqrMagnitude);
                    }
                    //each ball that is close to a pocket adds 0.1 to the score
                    float distanceSd = 0.5f;
                    score += Mathf.Min(0.1f * Mathf.Exp(-0.5f * minSqDist / (distanceSd * distanceSd)), 0.8f);
                    if (score >= 2)
                    {
                        score = 2;
                    }
                }
            }
        }


        float resultScore = score;
        PredictedShotScore = resultScore;
        RestoreState();

        return resultScore;
    }
}
