
/// <summary>
///   Common interface for weight constraints.
/// </summary>
/// 
public interface IWeightConstraint
{
    /// <summary>
    ///   Wires the constraint to the graph.
    /// </summary>
    /// 
    /// <param name="w">The weights tensor.</param>
    /// 
    /// <returns>The output tensor with the constraint applied.</returns>
    /// 
    Tensor Call(Tensor w);
}
