call D:\TCMXX\Tools\Anaconda3\Scripts\activate.bat
call activate ml-agents
tensorboard --logdir=%0\..\logs
call deactivate