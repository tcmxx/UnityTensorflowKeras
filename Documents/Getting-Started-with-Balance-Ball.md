Simple usage example:
Here is a simple example of how to use your existing UnityML agent and this repo to train neural netowkr using PPO in editor:

Copy you existing scene where you have implemented UnityML agent and Academy. Check Making a new Learning environment from UnityML's documentation.
You should have a Brain in your scene now. Change the BrainType to InternalTrainable. This option should be there if you have installed everything correctly.
Add a new GameObject, attach a script called TrainerPPO.cs to it. Assign this script to the Trainer field in your Brain.
Add a new GameObject, attach a script called RLModelPPO.cs to it. Assign this script to the ModelRef field in your TrainerPPO.
Create two scriptable objects: TrainerParamsPPO and RLNetworkSimpleAC. Assign those to your TrainerPPO's Parameters fields and your RLModelPPO's network fields respectively.
Click Play and it will start!
You can click Window/Grapher from menu to monitor your training process.(It is a modified version of old Grapher, when it was still free. It seems to be not free nor opensource anymore...I will remove it if it is causing any problem.)
