# Training a Generative Adversarial Network(GAN)

GAN is a type of commonly used neural network architecture that can to trained to generate desired data with variance. Some state-of-art GANs such as [Progressive GAN](https://www.youtube.com/watch?v=XOxxPcy5Gr4)
can generate really good fake data.

Here we only have only implemented the very basic conditional GAN that generates vector of data. Currently the GANs that can generate good images are normally not feasible to run
 on a computer without fancy GPUs.


(Document not finished yet..)
#### GANModel.cs
* `checkpointToLoad`: If you assign a model's saved checkpoint file to it, this will be loaded when model is initialized, regardless of the trainer's loading. Might be used when you are not using a trainer.
* `modelName`: The name of the model. It is used for the namescope When buliding the neural network. Can be empty by default.
* `weightSaveMode`: This decides the names of the weights of neural network when saving to checkpoints as serialized dictionary. No need to changes this ususally. 
* `Network`: You need to assign this field with a scriptable object that implements RLNetworkPPO.cs. 
* `generatorL2LossWeight`: L2 loss weight of the generator. Usually 0 is fine. 
* `outputShape`: Output shape of GAN. For ML-Agent, you can keep it unmodified, and the trainer will set it for you.
* `inputNoiseShape`: Input noise shape of GAN. Usually it is the same as the output shape.
* `inputConditionShape`: The input observation shape. For ML-Agent, you can keep it unmodified, and the trainer will set it for you.
* `generatorOptimizer`: The optimizer to use for this model to train generator.
* `discriminatorOptimizer`: The optimizer to use for this model to train discriminator.
* `initializeOnAwake`: Whether to initialize the GAN model on awake baed on shapes defined above. For ML-Agent environment, set this to false.

#### TrainerParamsGAN
See [TrainerParamsMimic](#trainerparamsmimic) for other parameters not listed below.
* `discriminatorTrainCount`: How many times the discriminator will be trained each training step.
* `generatorTrainCount`: How many times the generator will be trained each training step.
* `usePrediction`: Whether use [prediction method](https://www.semanticscholar.org/paper/Stabilizing-Adversarial-Nets-With-Prediction-Yadav-Shah/ec25504486d8751e00e613ca6fa64b256e3581c8) to stablize the training.
