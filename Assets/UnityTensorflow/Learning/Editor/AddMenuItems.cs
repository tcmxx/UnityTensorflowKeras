using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

public class AddMenuItems : MonoBehaviour {

    readonly static string checkpointPath = "UnityMLSaves";

    [MenuItem("GameObject/Unity In Game Machine Learning/MLAgents Integration/Create Supoervised Learning")]
    static void AddSLGO()
    {
        var obj1 = new GameObject("LearningModel_SL");
        obj1.AddComponent<SupervisedLearningModel>();
        var obj2 = new GameObject("Trainer_SL");
        obj2.AddComponent<TrainerMimic>();

        var obj3 = new GameObject("SL_Learning");
        obj1.transform.parent = obj3.transform;
        obj2.transform.parent = obj3.transform;

        //try to create parameter assets
        SupervisedLearningNetworkSimple network = null;
        TrainerParamsMimic trainerParam = null;
        CreateAssets<TrainerParamsMimic, SupervisedLearningNetworkSimple>("TrainerParamSL_" + obj1.scene.name + ".asset",
            "NetworkSL_" + obj1.scene.name + ".asset",
            out trainerParam, out network);
        network.hiddenLayers = new List<UnityNetwork.SimpleDenseLayerDef>();
        network.hiddenLayers.Add(new UnityNetwork.SimpleDenseLayerDef());

        var trainer = obj2.GetComponent<TrainerMimic>();
        trainer.modelRef = obj1.GetComponent<SupervisedLearningModel>();
        trainer.parameters = trainerParam;
        trainer.checkpointPath = checkpointPath;
        trainer.checkpointFileName = "Checkpoint_" + obj1.scene.name + ".bytes";
        trainer.trainingDataSaveFileName = "Collected_SL_Data_" + obj1.scene.name + ".bytes";

        ((SupervisedLearningModel)trainer.modelRef).network = network;
    }

    [MenuItem("GameObject/Unity In Game Machine Learning/MLAgents Integration/Create PPO")]
    static void AddPPOGO()
    {
        var obj1 = new GameObject("LearningModel_PPO");
        obj1.AddComponent<RLModelPPO>();
        var obj2 = new GameObject("Trainer_PPO");
        obj2.AddComponent<TrainerPPO>();

        var obj3 = new GameObject("PPO_Learning");
        obj1.transform.parent = obj3.transform;
        obj2.transform.parent = obj3.transform;

        //try to create parameter assets
        RLNetworkSimpleAC network = null;
        TrainerParamsPPO trainerParam = null;
        CreateAssets<TrainerParamsPPO, RLNetworkSimpleAC>("TrainerParamPPO_" + obj1.scene.name + ".asset",
            "NetworkPPO_" + obj1.scene.name + ".asset",
            out trainerParam, out network);
        network.actorHiddenLayers = new List<UnityNetwork.SimpleDenseLayerDef>();
        network.actorHiddenLayers.Add(new UnityNetwork.SimpleDenseLayerDef());
        network.criticHiddenLayers = new List<UnityNetwork.SimpleDenseLayerDef>();
        network.criticHiddenLayers.Add(new UnityNetwork.SimpleDenseLayerDef());

        var trainer = obj2.GetComponent<TrainerPPO>();
        trainer.modelRef = obj1.GetComponent<RLModelPPO>();
        trainer.parameters = trainerParam;
        trainer.checkpointPath = checkpointPath;
        trainer.checkpointFileName = "Checkpoint_" + obj1.scene.name + ".bytes";

        ((RLModelPPO)trainer.modelRef).network = network;
    }

    
    [MenuItem("GameObject/Unity In Game Machine Learning/MLAgents Integration/Create PPOCMA")]
    static void AddPPOCMAGO()
    {
        var obj1 = new GameObject("LearningModel_PPOCMA");
        obj1.AddComponent<RLModelPPOCMA>();
        var obj2 = new GameObject("Trainer_PPOCMA");
        obj2.AddComponent<TrainerPPOCMA>();

        var obj3 = new GameObject("PPOCMA_Learning");
        obj1.transform.parent = obj3.transform;
        obj2.transform.parent = obj3.transform;

        //try to create parameter assets
        RLNetworkACSeperateVar network = null;
        TrainerParamsPPO trainerParam = null;
        CreateAssets<TrainerParamsPPO, RLNetworkACSeperateVar>("TrainerParamPPOCMA_" + obj1.scene.name + ".asset",
            "NetworkPPOCMA_" + obj1.scene.name + ".asset",
            out trainerParam, out network);
        network.actorHiddenLayers = new List<UnityNetwork.SimpleDenseLayerDef>();
        network.actorHiddenLayers.Add(new UnityNetwork.SimpleDenseLayerDef());
        network.criticHiddenLayers = new List<UnityNetwork.SimpleDenseLayerDef>();
        network.criticHiddenLayers.Add(new UnityNetwork.SimpleDenseLayerDef());

        var trainer = obj2.GetComponent<TrainerPPOCMA>();
        trainer.modelRef = obj1.GetComponent<RLModelPPOCMA>();
        trainer.parameters = trainerParam;
        trainer.checkpointPath = checkpointPath;
        trainer.checkpointFileName = "Checkpoint_" + obj1.scene.name + ".bytes";

        ((RLModelPPOCMA)trainer.modelRef).network = network;
    }



    [MenuItem("GameObject/Unity In Game Machine Learning/MLAgents Integration/Create Neural Evolution")]
    static void AddNEGO()
    {
        var obj1 = new GameObject("LearningModel_NE");
        obj1.AddComponent<SupervisedLearningModel>();
        var obj2 = new GameObject("Trainer_NE");
        obj2.AddComponent<TrainerNeuralEvolution>();

        var obj3 = new GameObject("NE_Learning");
        obj1.transform.parent = obj3.transform;
        obj2.transform.parent = obj3.transform;

        //try to create parameter assets
        TrainerParamsNeuralEvolution trainerParam = null;
        SupervisedLearningNetworkSimple network = null;
        CreateAssets<TrainerParamsNeuralEvolution, SupervisedLearningNetworkSimple>("TrainerParamNE_" + obj1.scene.name + ".asset",
            "NetworkNE_" + obj1.scene.name + ".asset",
            out trainerParam, out network);
        network.hiddenLayers = new List<UnityNetwork.SimpleDenseLayerDef>();
        network.hiddenLayers.Add(new UnityNetwork.SimpleDenseLayerDef());


        var trainer = obj2.GetComponent<TrainerNeuralEvolution>();
        trainer.modelRef = obj1.GetComponent<SupervisedLearningModel>();
        trainer.parameters = trainerParam;
        trainer.checkpointPath = checkpointPath;
        trainer.checkpointFileName = "Checkpoint_" + obj1.scene.name + ".bytes";
        trainer.evolutionDataSaveFileName = "NeuralEvolutionData_" + obj1.scene.name + ".bytes";

        ((SupervisedLearningModel)trainer.modelRef).network = network;
    }

    [MenuItem("GameObject/Unity In Game Machine Learning/MLAgents Integration/Create GAN with Supervised Learning")]
    static void AddGAN()
    {
        //ccreate the GOs
        var obj1 = new GameObject("LearningModel_GAN");
        obj1.AddComponent<GANModel>();
        var obj2 = new GameObject("Trainer_GAN");
        obj2.AddComponent<TrainerMimic>();

        var obj3 = new GameObject("GAN_Learning");
        obj1.transform.parent = obj3.transform;
        obj2.transform.parent = obj3.transform;

        //try to create parameter assets
        TrainerParamsGAN trainerParam = null;
        GANNetworkDense network = null;
        CreateAssets<TrainerParamsGAN, GANNetworkDense>("TrainerParamGAN_" + obj1.scene.name + ".asset",
            "NetworkGAN_" + obj1.scene.name + ".asset",
            out trainerParam, out network);
        network.discriminatorHiddenLayers = new List<UnityNetwork.SimpleDenseLayerDef>();
        network.discriminatorHiddenLayers.Add(new UnityNetwork.SimpleDenseLayerDef());

        var trainer = obj2.GetComponent<TrainerMimic>();
        trainer.modelRef = obj1.GetComponent<GANModel>();
        trainer.parameters = trainerParam;
        trainer.checkpointPath = checkpointPath;
        trainer.checkpointFileName = "Checkpoint_" + obj1.scene.name + ".bytes";
        trainer.trainingDataSaveFileName = "Collected_SL_Data_" + obj1.scene.name + ".bytes";

        ((GANModel)trainer.modelRef).network = network;


    }



    public static void CreateAssets<T1,T2>(string name1, string name2, out T1 a1, out T2 a2) where T1 : ScriptableObject where T2 : ScriptableObject
    {
        string path = EditorUtility.SaveFolderPanel("Select a directory under Asset/ to Create Parameter Assets", "Assets", "");
        a1 = null;
        a2 = null;

        var startIndex = path.IndexOf("Assets");
        path = path.Substring(startIndex);
        if (path.Length == 0)
        {
            Debug.LogWarning("No Training Asset is created automatically. Please create them and assign them to Trainer and LearningModel.");
        }
        else
        {
            a1 = CreateAsset<T1>(path + Path.DirectorySeparatorChar + name1);
            a2 = CreateAsset<T2>(path + Path.DirectorySeparatorChar + name2);
        }
    }

    /// <summary>
    //	This makes it easy to create, name and place unique new ScriptableObject asset files.
    /// </summary>
    public static T CreateAsset<T>(string pathAndName) where T : ScriptableObject
    {
        T asset = ScriptableObject.CreateInstance<T>();

        string assetPathAndName = AssetDatabase.GenerateUniqueAssetPath(pathAndName);

        AssetDatabase.CreateAsset(asset, assetPathAndName);

        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();
        EditorUtility.FocusProjectWindow();

        return asset;
    }
}
