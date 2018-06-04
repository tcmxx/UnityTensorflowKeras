
    using System.Collections.Generic;

    public interface IOptimizer
    {
        List<List<Tensor>> get_updates(List<Tensor> collected_trainable_weights, Dictionary<Tensor, IWeightConstraint> constraints, Tensor total_loss);
    }
