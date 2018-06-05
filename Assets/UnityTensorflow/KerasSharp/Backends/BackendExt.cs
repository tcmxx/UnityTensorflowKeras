using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class BackendExt {

    public static Tensor normal_probability(this IBackend b, Tensor input, Tensor mean, Tensor variance)
    {
        //probability
        var diff = input - mean;
        var temp1 = diff * diff;
        temp1 = temp1 / (2 * variance);
        temp1 = b.exp(0 - temp1);

        var temp2 = 1.0f / b.square((2 * Mathf.PI) * variance);
        return temp1 * temp2;
    }
}
