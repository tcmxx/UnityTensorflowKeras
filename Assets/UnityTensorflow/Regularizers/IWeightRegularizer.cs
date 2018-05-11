﻿// Keras-Sharp: C# port of the Keras library
// https://github.com/cesarsouza/keras-sharp
//
// Based under the Keras library for Python. See LICENSE text for more details.
//
//    The MIT License(MIT)
//    
//    Permission is hereby granted, free of charge, to any person obtaining a copy
//    of this software and associated documentation files (the "Software"), to deal
//    in the Software without restriction, including without limitation the rights
//    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//    copies of the Software, and to permit persons to whom the Software is
//    furnished to do so, subject to the following conditions:
//    
//    The above copyright notice and this permission notice shall be included in all
//    copies or substantial portions of the Software.
//    
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//    SOFTWARE.
//

//  modified from KerasSharp

using System.Collections.Generic;
using System.Linq;

/// <summary>
///   Common interface for weight regularizers.
/// </summary>
/// 
public interface IWeightRegularizer
{
    /// <summary>
    /// Wires the regularizer to the graph.
    /// </summary>
    /// 
    /// <param name="w">The weights tensor.</param>
    /// 
    /// <returns>The output tensor with the regularization applied.</returns>
    /// 
    UnityTFTensor Call(UnityTFTensor input);

    /// <summary>
    /// Wires the regularizer to the graph.
    /// </summary>
    /// 
    /// <param name="w">The weights tensor.</param>
    /// 
    /// <returns>The output tensor with the regularization applied.</returns>
    /// 
    List<UnityTFTensor> Call(List<UnityTFTensor> input);
}

public abstract class WeightRegularizerBase
{
    public List<UnityTFTensor> Call(List<UnityTFTensor> input)
    {
        return input.Select(x => Call(x)).ToList();
    }

    public abstract UnityTFTensor Call(UnityTFTensor input);

}
