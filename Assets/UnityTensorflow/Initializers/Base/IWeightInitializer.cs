
    /// <summary>
    ///   Common interface for weight initialization strategies.
    /// </summary>
    /// 
    public interface IWeightInitializer
    {
        /// <summary>
        ///   Creates a <see cref="TFTensor"/> with the desired initial weights.
        /// </summary>
        /// 
        /// <param name="shape">The shape of the tensor to be generated.</param>
        /// <param name="dtype">The <see cref="TFDataType">data type</see> of the tensor to be generated.</param>
        /// 
        /// <returns>A <see cref="TFTensor"/> initialized of dimensions <paramref name="shape"/>
        ///   and element data type <paramref name="dtype"/> that has been initialized using this
        ///   strategy.</returns>
        /// 
        UnityTFTensor Call(int[] shape, DataType? dtype = null);
    }
