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

//  modeified from kerassharp

using System.Collections.Generic;

/// <summary>
///   Base class for <see cref="IActivationFunction"/> implementations.
/// </summary>
/// 
public abstract class ActivationFunctionBase
{
    /// <summary>
    /// Wires the activation function to the graph.
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <returns>The output tensor with the activation function applied.</returns>
    public abstract UnityTFTensor Call(UnityTFTensor inputs, UnityTFTensor mask);

    /// <summary>
    /// Wires the activation function to the graph.
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <returns>The output tensor with the activation function applied.</returns>
    public List<UnityTFTensor> Call(List<UnityTFTensor> inputs, List<UnityTFTensor> mask)
    {
        List<UnityTFTensor> output = new List<UnityTFTensor>();
        for (int i = 0; i < inputs.Count; i++)
            output.Add(Call(inputs[i], mask != null ? mask[i] : null));
        return output;
    }

}
