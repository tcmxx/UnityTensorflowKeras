call D:\TCMXX\Tools\Anaconda3\Scripts\activate.bat
call activate tensorflow
tensorboard --logdir=%0\..\logs
call deactivate