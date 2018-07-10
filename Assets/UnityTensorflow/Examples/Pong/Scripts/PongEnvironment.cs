using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PongEnvironment : MonoBehaviour
{

    public PongAgent leftAgent;
    public PongAgent rightAgent;

    public float failureReward = -1;
    public float winReward = 1;
    public float hitBallReward = 0.1f;

    public float racketSpeed = 0.02f;
    public float ballSpeed = 0.01f;
    public float racketWidth = 0.05f;

    public readonly int ActionUp = 2;
    public readonly int ActionDown = 0;
    public readonly int ActionStay = 1;

    public float leftStartX = -1;
    public float rightStartX = 1;
    public Vector2 arenaSize = new Vector2(2.2f, 1.0f);


    [Header("Informations")]
    [ReadOnly]
    [SerializeField]
    protected float leftHitOrMiss = 0;
    [ReadOnly]
    [SerializeField]
    protected float rightHitOrMiss = 0;

    public GameState CurrentGameState { get { return currentGameState; } }
    private GameState currentGameState;

    protected int step = 0;

    public struct GameState
    {
        public Vector2 ballVelocity;
        public Vector2 ballPosition;
        public float leftY;
        public float rightY;
    }

    private void Awake()
    {
        Debug.Assert(leftAgent != null && rightAgent != null, "Please set agentLeft and agentRight for this environment");
        leftAgent.environment = this;
        rightAgent.environment = this;
    }

    private void Start()
    {
        Reset();
    }


    public float[] CurrentState(PongAgent actor)
    {
        float[] result = null;
        if (actor == leftAgent)
        {
            result = new float[] {
                currentGameState.leftY,
                currentGameState.rightY,
                currentGameState.ballPosition.x,
                currentGameState.ballPosition.y,
                currentGameState.ballVelocity.x,
                currentGameState.ballVelocity.y
            };
        }
        else if(actor == rightAgent)
        {
            result = new float[] {
                currentGameState.rightY,
                currentGameState.leftY,
                -currentGameState.ballPosition.x,
                currentGameState.ballPosition.y,
                -currentGameState.ballVelocity.x,
                currentGameState.ballVelocity.y
            };
        }
        else
        {
            Debug.LogError("Wrong agent");
        }
        return result;
    }


    private void FixedUpdate()
    {
        Step();
    }


    public void MoveRacket(PongAgent agent, float action)
    {
        int actionInt = Mathf.RoundToInt(action);

        Debug.Assert(actionInt >= ActionDown && actionInt < ActionUp + 1);

        if (agent == leftAgent)
        {
            //move the rackets
            currentGameState.leftY += racketSpeed * (actionInt - 1);
            currentGameState.leftY = Mathf.Clamp(currentGameState.leftY, -arenaSize.y / 2 + racketWidth / 2, arenaSize.y / 2 - racketWidth / 2);
        }
        else if(agent == rightAgent)
        {
            currentGameState.rightY += racketSpeed * (actionInt - 1);
            currentGameState.rightY = Mathf.Clamp(currentGameState.rightY, -arenaSize.y / 2 + racketWidth / 2, arenaSize.y / 2 - racketWidth / 2);
        }
        else
        {
            Debug.LogError("Wrong agent");
        }
    }


    protected void Step()
    {
        //move the ball
        Vector2 oldBallPosition = currentGameState.ballPosition;
        currentGameState.ballPosition += currentGameState.ballVelocity;

        //detect collision of ball with wall
        Vector2 newBallVel = currentGameState.ballVelocity;
        if (currentGameState.ballPosition.y > arenaSize.y / 2 || currentGameState.ballPosition.y < -arenaSize.y / 2)
        {
            newBallVel.y = -newBallVel.y;

        }
        if (currentGameState.ballPosition.x > arenaSize.x / 2)
        {
            leftAgent.AddReward( winReward);
            rightAgent .AddReward( failureReward);
            leftAgent.Done();
            rightAgent.Done();
        }
        else if (currentGameState.ballPosition.x < -arenaSize.x / 2)
        {
            leftAgent.AddReward(failureReward);
            rightAgent.AddReward(winReward);
            leftAgent.Done();
            rightAgent.Done();
        }

        //detect collision of the ball with the rackets
        if (currentGameState.ballPosition.x < leftStartX && oldBallPosition.x > leftStartX)
        {
            Vector2 moveVector = (currentGameState.ballPosition - oldBallPosition);
            float yHit = (moveVector * Mathf.Abs((oldBallPosition.x - leftStartX) / moveVector.x) + oldBallPosition).y;
            float yHitRatio = (currentGameState.leftY - yHit) / (racketWidth / 2);
            if (Mathf.Abs(yHitRatio) < 1)
            {
                //hit the left racket
                newBallVel.x = -newBallVel.x;
                newBallVel.y = -Mathf.Abs(newBallVel.x) * yHitRatio * 2;
                newBallVel = newBallVel.normalized * ballSpeed;
                leftAgent.AddReward(hitBallReward);
                leftHitOrMiss = 1;
            }
            else
            {
                leftHitOrMiss = -1;
            }
        }
        else if (currentGameState.ballPosition.x > rightStartX && oldBallPosition.x < rightStartX)
        {
            Vector2 moveVector = (currentGameState.ballPosition - oldBallPosition);
            float yHit = (moveVector * Mathf.Abs((oldBallPosition.x - rightStartX) / moveVector.x) + oldBallPosition).y;
            float yHitRatio = (currentGameState.rightY - yHit) / (racketWidth / 2);
            if (Mathf.Abs(yHitRatio) < 1)
            {
                //hit the right racket
                newBallVel.x = -newBallVel.x;
                newBallVel.y = -Mathf.Abs(newBallVel.x) * yHitRatio * 2;
                newBallVel = newBallVel.normalized * ballSpeed;
                rightAgent.AddReward(hitBallReward);
                rightHitOrMiss = 1;
            }
            else
            {
                rightHitOrMiss = -1;
            }
        }
        else
        {
            leftHitOrMiss = 0;
            rightHitOrMiss = 0;
        }

        //update the velocity
        currentGameState.ballVelocity = newBallVel;


        step++;
    }

    // to be implemented by the developer
    public void Reset()
    {
        currentGameState.leftY = 0;
        currentGameState.rightY = 0;
        currentGameState.ballPosition = Vector2.zero;
        Vector2 initialVel = Random.insideUnitCircle;
        if (Mathf.Abs(initialVel.y) > Mathf.Abs(initialVel.x))
        {
            float temp = initialVel.y;
            initialVel.y = initialVel.x;
            initialVel.x = temp;
        }
        currentGameState.ballVelocity = initialVel.normalized * ballSpeed;
        step = 0;
    }

}
