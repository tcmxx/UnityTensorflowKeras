
    using System.Collections.Generic;

    public interface IOptimizer
    {
        List<List<UnityTFTensor>> get_updates(List<UnityTFTensor> collected_trainable_weights, Dictionary<UnityTFTensor, IWeightConstraint> constraints, UnityTFTensor total_loss);
    }
