using System.Collections;
using System.Collections.Generic;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

using System.Linq;

/// CoreBrain which decides actions using internally embedded TensorFlow model.
public class CoreBrainInternalTrainable : ScriptableObject, CoreBrain
{

    bool hasRecurrent;

    /// Reference to the brain that uses this CoreBrainInternal
    public Brain brain;
    public Trainer trainer;

    public bool trainModel = true;  //whether to train the model

    private Dictionary<Agent, AgentInfo> currentInfo;
    private TakeActionOutput prevActionOutput;


    /// Create the reference to the brain
    public void SetBrain(Brain b)
    {
        brain = b;
    }

    /// Loads the tensorflow graph model to generate a TFGraph object
    public void InitializeCoreBrain(MLAgents.Batcher brainBatcher)
    {
    }



    /// Uses the stored information to run the tensorflow graph and generate 
    /// the actions.
    public void DecideAction(Dictionary<Agent, AgentInfo> agentInfo)
    {

        
        trainer.AddExperience(currentInfo, agentInfo, prevActionOutput);
        trainer.ProcessExperience(currentInfo, agentInfo);

        if (trainer.IsReadyUpdate() && trainModel && trainer.GetStep() <= trainer.GetStep())
        {
            trainer.UpdateModel();
        }

        trainer.WriteSummary();

        if (trainModel && trainer.GetStep() <= trainer.GetStep())
        {
            trainer.IncrementStep();
            trainer.UpdateLastReward();
        }

        //TODO: Save the model every certain amount of steps

        //shallow copy the info
        currentInfo = new Dictionary<Agent, AgentInfo>(agentInfo);


        prevActionOutput = trainer.TakeAction(currentInfo);

        //TODO Update the agent action
        List<Agent> agentList = agentInfo.Keys.ToList();
        if (hasRecurrent)
        {
            foreach (Agent agent in agentList)
            {
                //agent.UpdateMemoriesAction(prevActionOutput.memory[agent].ToList());
            }
        }


        foreach (Agent agent in agentList)
        {
            agent.UpdateVectorAction(prevActionOutput.outputAction[agent]);

        }

    }

    /// Displays the parameters of the CoreBrainInternal in the Inspector 
    public void OnInspector()
    {
#if UNITY_EDITOR
        EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);

        var serializedBrain = new SerializedObject(this);



        var trainerProperty = serializedBrain.FindProperty("trainer");
        serializedBrain.Update();
        EditorGUILayout.PropertyField(trainerProperty, true);
        serializedBrain.ApplyModifiedProperties();
#endif
    }

    /// <summary>
    /// Converts a list of Texture2D into a Tensor.
    /// </summary>
    /// <returns>
    /// A 4 dimensional float Tensor of dimension
    /// [batch_size, height, width, channel].
    /// Where batch_size is the number of input textures,
    /// height corresponds to the height of the texture,
    /// width corresponds to the width of the texture,
    /// channel corresponds to the number of channels extracted from the
    /// input textures (based on the input blackAndWhite flag
    /// (3 if the flag is false, 1 otherwise).
    /// The values of the Tensor are between 0 and 1.
    /// </returns>
    /// <param name="textures">
    /// The list of textures to be put into the tensor.
    /// Note that the textures must have same width and height.
    /// </param>
    /// <param name="blackAndWhite">
    /// If set to <c>true</c> the textures
    /// will be converted to grayscale before being stored in the tensor.
    /// </param>
    public static float[,,,] BatchVisualObservations(
        List<Texture2D> textures, bool blackAndWhite)
    {
        int batchSize = textures.Count();
        int width = textures[0].width;
        int height = textures[0].height;
        int pixels = 0;
        if (blackAndWhite)
            pixels = 1;
        else
            pixels = 3;
        float[,,,] result = new float[batchSize, height, width, pixels];

        for (int b = 0; b < batchSize; b++)
        {
            Color32[] cc = textures[b].GetPixels32();
            for (int w = 0; w < width; w++)
            {
                for (int h = 0; h < height; h++)
                {
                    Color32 currentPixel = cc[h * width + w];
                    if (!blackAndWhite)
                    {
                        // For Color32, the r, g and b values are between
                        // 0 and 255.
                        result[b, textures[b].height - h - 1, w, 0] =
                            currentPixel.r / 255.0f;
                        result[b, textures[b].height - h - 1, w, 1] =
                            currentPixel.g / 255.0f;
                        result[b, textures[b].height - h - 1, w, 2] =
                            currentPixel.b / 255.0f;
                    }
                    else
                    {
                        result[b, textures[b].height - h - 1, w, 0] =
                            (currentPixel.r + currentPixel.g + currentPixel.b)
                            / 3;
                    }
                }
            }
        }
        return result;
    }

}
