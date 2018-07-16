using MLAgents;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(MazeViewer))]
public class MazeAgent : Agent
{

    public Vector2Int mazeDimension;
    public bool regenerateMapOnReset = false;
    public bool randomWallChance = false;
    public bool noVectorObservation = true;
    public float wallChanceOnNonPath = 0.3f;
    public int maxStepAllowed = 20;
    public float failureReward = -100;
    public float maxWinReward = 100;
    public float goToWallReward;
    public float goUpReward = 1;
    public float goCloserReward = 1;
    public float stepCostReward = -0.1f;

    private Vector2Int startPosition;
    private Vector2Int goalPosition;
    private Vector2Int currentPlayerPosition;
    public bool Win { get; private set; }
    public float[,] map;
    private Dictionary<int, GameState> savedState;

    private MazeViewer viewer;

    [Header("Info")]
    [ReadOnly]
    [SerializeField]
    protected int steps = 0;

    public readonly int WallInt = 0;
    public readonly int PlayerInt = 2;
    public readonly int PathInt = 1;
    public readonly int GoalInt = 3;

    private struct GameState
    {
        public float[,] map;
        public Vector2Int startPosition;
        public Vector2Int goalPosition;
        public Vector2Int currentPlayerPosition;
        public bool win;
    }

    private void Awake()
    {
        viewer = GetComponent<MazeViewer>();
    }
    // Use this for initialization
    public override void InitializeAgent()
    {
        savedState = new Dictionary<int, GameState>();
        viewer.InitializeGraphic(this);
        AgentReset();
    }

    public override void CollectObservations()
    {
        if (noVectorObservation)
            return;

        float[] result = new float[mazeDimension.x * mazeDimension.y];
        for (int x = 0; x < mazeDimension.x; ++x)
        {
            for (int y = 0; y < mazeDimension.y; ++y)
            {
                result[y + x * mazeDimension.y] = (map[x, y] - 1.5f) / 1.5f;
            }
        }
        AddVectorObs(result);
    }

    public override void AgentReset()
    {
        steps = 0;
        if (regenerateMapOnReset)
        {
            RegenerateMap();
            SaveState(-1);
        }
        else
        {
            LoadState(-1);
        }
    }

    /// <summary>
    /// take a action and return the reward
    /// </summary>
    /// <param name="action">0 left 1 right 2 down 3 up</param>
    /// <returns>reward of this action</returns>
    public override void AgentAction(float[] vectorAction, string textAction)
    {
        int action = Mathf.RoundToInt(vectorAction[0]);
        float returnReward = 0;
        //calculate the distance to goal before the action
        int distanceBefore = Mathf.Abs((currentPlayerPosition - goalPosition).x) + Mathf.Abs((currentPlayerPosition - goalPosition).y);

        Vector2Int toPosition = currentPlayerPosition;
        //do the action
        switch (action)
        {
            case 0:
                toPosition.x -= 1;
                break;
            case 1:
                toPosition.x += 1;
                break;
            case 2:
                toPosition.y -= 1;
                break;
            case 3:
                toPosition.y += 1;
                break;
            default:
                Debug.LogError("invalid action number");
                break;
        }

        bool reachGoal;
        float stepChangedReward;
        StepFromTo(currentPlayerPosition, toPosition, out stepChangedReward, out reachGoal);
        returnReward += stepChangedReward;

        //reward for move closer to the destination
        int distanceAfter = Mathf.Abs((currentPlayerPosition - goalPosition).x) + Mathf.Abs((currentPlayerPosition - goalPosition).y);
        if (distanceAfter < distanceBefore)
        {
            returnReward += goCloserReward;
        }

        //reward for going up
        if (action == 3)
        {
            returnReward += goUpReward;
        }
        
        if (reachGoal)
        {
            Done();
            Win = true;
        }
        if (steps >= maxStepAllowed)
        {
            Done();
            Win = false;
            returnReward += failureReward;
        }

        steps++;
        AddReward(returnReward);

        viewer.UpdateGraphics(this);
    }


    public void SaveState(int key)
    {
        float[,] copiedMap = new float[mazeDimension.x, mazeDimension.y];
        System.Buffer.BlockCopy(map, 0, copiedMap, 0, map.Length * sizeof(float));

        GameState state = new GameState();
        state.map = copiedMap;
        state.currentPlayerPosition = currentPlayerPosition;
        state.goalPosition = goalPosition;
        state.startPosition = startPosition;
        state.win = Win;
        savedState[key] = state;
    }


    public bool LoadState(int key)
    {
        MazeAgent fromEnv = this;
        if (fromEnv.savedState.ContainsKey(key))
        {
            GameState state = fromEnv.savedState[key];
            System.Buffer.BlockCopy(state.map, 0, map, 0, map.Length * sizeof(float));
            currentPlayerPosition = state.currentPlayerPosition;
            goalPosition = state.goalPosition;
            startPosition = state.startPosition;
            Win = state.win;
            viewer.UpdateGraphics(this);
            return true;
        }
        else
        {
            return false;
        }

        
    }





    private void RegenerateMap()
    {
        map = new float[mazeDimension.x, mazeDimension.y];
        GeneratePossiblePath();
        GenerateExtraPath();

        viewer.UpdateGraphics(this);
    }

    //mark a path with true. The generator will guarantee that this path is walkable
    private void GeneratePossiblePath()
    {


        int place = Random.Range(0, mazeDimension.x);
        int prevPlace = place;
        map[place, mazeDimension.y - 1] = GoalInt;
        goalPosition = new Vector2Int(place, mazeDimension.y - 1);

        bool toggle = true;
        for (int i = mazeDimension.y - 2; i >= 0; --i)
        {
            if (toggle)
            {
                toggle = false;
                map[prevPlace, i] = PathInt;
            }
            else
            {
                toggle = true;
                place = Random.Range(0, mazeDimension.x);
                for (int j = Mathf.Min(place, prevPlace); j <= Mathf.Max(place, prevPlace); ++j)
                {
                    map[j, i] = PathInt;
                }
            }
            prevPlace = place;
        }

        startPosition = new Vector2Int(place, 0);
        map[place, 0] = PlayerInt;
        currentPlayerPosition = startPosition;
    }

    private void GenerateExtraPath()
    {
        float wallChance = wallChanceOnNonPath;
        if (randomWallChance)
        {
            wallChance = Random.Range(0.0f, 1.0f);
        }
        for (int i = 0; i < mazeDimension.x; ++i)
        {
            for (int j = 0; j < mazeDimension.y; ++j)
            {
                if (map[i, j] == WallInt && Random.Range(0.0f, 1.0f) > wallChance)
                {
                    map[i, j] = PathInt;
                }
            }
        }
    }

    private void StepFromTo(Vector2Int from, Vector2Int to, out float stepChangedReward, out bool reachedGoal)
    {
        Debug.Assert(map[from.x, from.y] == PlayerInt && currentPlayerPosition.Equals(from));
        stepChangedReward = 0;
        stepChangedReward += stepCostReward;
        if (to.x < 0 || to.y < 0 || to.x >= mazeDimension.x || to.y >= mazeDimension.y)
        {
            //run to the edge
            stepChangedReward += goToWallReward;
            reachedGoal = false;
        }
        else
        {
            if (map[to.x, to.y] == WallInt)
            {
                //run into a wall
                //run to the edge
                stepChangedReward += goToWallReward;
                reachedGoal = false;
            }
            else if (map[to.x, to.y] == GoalInt)
            {
                //reach the goal
                stepChangedReward += maxWinReward;
                reachedGoal = true;
            }
            else
            {
                //move successfully
                map[currentPlayerPosition.x, currentPlayerPosition.y] = PathInt;
                currentPlayerPosition = to;
                map[currentPlayerPosition.x, currentPlayerPosition.y] = PlayerInt;
                reachedGoal = false;
            }
        }
    }
}
