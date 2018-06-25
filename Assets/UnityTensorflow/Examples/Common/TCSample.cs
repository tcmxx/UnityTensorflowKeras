using System.Collections;
using System.Collections.Generic;
using UnityEngine;
//using System;




namespace TCUtils{


/// collection of some useful sampling/random methods








	/// Poisson-disc sampling using Bridson's algorithm.
	/// Adapted from Mike Bostock's Javascript source: http://bl.ocks.org/mbostock/19168c663618b7f07158
	///
	/// See here for more information about this algorithm:
	///   http://devmag.org.za/2009/05/03/poisson-disk-sampling/
	///   http://bl.ocks.org/mbostock/dbb02448b0f93e4c82c3
	///
	/// Usage:
	///   PoissonDiscSampler sampler = new PoissonDiscSampler(10, 5, 0.3f);
	///   foreach (Vector2 sample in sampler.Samples()) {
	///       // ... do something, like instantiate an object at (sample.x, sample.y) for example:
	///       Instantiate(someObject, new Vector3(sample.x, 0, sample.y), Quaternion.identity);
	///   }
	///
	/// Author: Gregory Schlomoff (gregory.schlomoff@gmail.com)
	/// Released in the public domain
	public class PoissonDiscSampler
	{
		private const int k = 30;  // Maximum number of attempts before marking a sample as inactive.

		private readonly Rect rect;
		private readonly float radius2;  // radius squared
		private readonly float cellSize;
		private Vector2[,] grid;
		public List<Vector2> ActiveSamples { get; private set; } = new List<Vector2>();

		/// Create a sampler with the following parameters:
		///
		/// width:  each sample's x coordinate will be between [0, width]
		/// height: each sample's y coordinate will be between [0, height]
		/// radius: each sample will be at least `radius` units away from any other sample, and at most 2 * `radius`.
		public PoissonDiscSampler(float width, float height, float radius)
		{
			rect = new Rect(0, 0, width, height);
			radius2 = radius * radius;
			cellSize = radius / Mathf.Sqrt(2);
			grid = new Vector2[Mathf.CeilToInt(width / cellSize),
				Mathf.CeilToInt(height / cellSize)];
		}

		/// Return a lazy sequence of samples. You typically want to call this in a foreach loop, like so:
		///   foreach (Vector2 sample in sampler.Samples()) { ... }
		public IEnumerable<Vector2> Samples()
		{
			// First sample is choosen randomly
			yield return AddSample(new Vector2(Random.value * rect.width, Random.value * rect.height));

			while (ActiveSamples.Count > 0) {

				// Pick a random active sample
				int i = (int) Random.value * ActiveSamples.Count;
				Vector2 sample = ActiveSamples[i];

				// Try `k` random candidates between [radius, 2 * radius] from that sample.
				bool found = false;
				for (int j = 0; j < k; ++j) {

					float angle = 2 * Mathf.PI * Random.value;
					float r = Mathf.Sqrt(Random.value * 3 * radius2 + radius2); // See: http://stackoverflow.com/questions/9048095/create-random-number-within-an-annulus/9048443#9048443
					Vector2 candidate = sample + r * new Vector2(Mathf.Cos(angle), Mathf.Sin(angle));

					// Accept candidates if it's inside the rect and farther than 2 * radius to any existing sample.
					if (rect.Contains(candidate) && IsFarEnough(candidate)) {
						found = true;
						yield return AddSample(candidate);
						break;
					}
				}

				// If we couldn't find a valid candidate after k attempts, remove this sample from the active samples queue
				if (!found) {
					ActiveSamples[i] = ActiveSamples[ActiveSamples.Count - 1];
					ActiveSamples.RemoveAt(ActiveSamples.Count - 1);
				}
			}
		}

		private bool IsFarEnough(Vector2 sample)
		{
			GridPos pos = new GridPos(sample, cellSize);

			int xmin = Mathf.Max(pos.x - 2, 0);
			int ymin = Mathf.Max(pos.y - 2, 0);
			int xmax = Mathf.Min(pos.x + 2, grid.GetLength(0) - 1);
			int ymax = Mathf.Min(pos.y + 2, grid.GetLength(1) - 1);

			for (int y = ymin; y <= ymax; y++) {
				for (int x = xmin; x <= xmax; x++) {
					Vector2 s = grid[x, y];
					if (s != Vector2.zero) {
						Vector2 d = s - sample;
						if (d.x * d.x + d.y * d.y < radius2) return false;
					}
				}
			}

			return true;

			// Note: we use the zero vector to denote an unfilled cell in the grid. This means that if we were
			// to randomly pick (0, 0) as a sample, it would be ignored for the purposes of proximity-testing
			// and we might end up with another sample too close from (0, 0). This is a very minor issue.
		}

		/// Adds the sample to the active samples queue and the grid before returning it
		private Vector2 AddSample(Vector2 sample)
		{
			ActiveSamples.Add(sample);
			GridPos pos = new GridPos(sample, cellSize);
			grid[pos.x, pos.y] = sample;
			return sample;
		}

		/// Helper struct to calculate the x and y indices of a sample in the grid
		private struct GridPos
		{
			public int x;
			public int y;

			public GridPos(Vector2 sample, float cellSize)
			{
				x = (int)(sample.x / cellSize);
				y = (int)(sample.y / cellSize);

			}
		}
	}





	/// <summary>
	/// Low discrepancy sequences. 
	/// </summary>
	public static class LowDiscrepancySequence{


		/// <summary>
		/// Generates the hammersley2 d.
		/// </summary>
		/// <returns>The hammersley2 d.</returns>
		/// <param name="n"> number of point in the sequenct</param>
		/// <param name="p">base for generating the sequence</param>
		public static List<Vector2> GenerateHammersley2D(int n, int p) {
			List<Vector2> points = new List<Vector2>(n);

			for (int k = 0; k < n; ++k) {
				float tempP = p;
				int tempK = k;
				float tempY = 0;
				while (tempK > 0) {
					float a = tempK%p;
					tempY = tempY + a / tempP;
					tempK = tempK / p;
					tempP = tempP * p;
				}
				points[k]= new Vector2((float)k / n, tempY);
			}
			return points;
		}


		/// <summary>
		/// Generates one halton number for index i and base b
		/// See https://en.wikipedia.org/wiki/Halton_sequence
		/// </summary>
		/// <returns>The halton.</returns>
		/// <param name="i">The index.</param>
		/// <param name="b">The base</param>
		public static float GenerateHalton(int i, int b){
			float f = 1;
			float r = 0;

			while(i > 0){
				f = f/b;
				r = r + f * (i % b);
				i = i/b;
			}
			return r;

		}


		/// <summary>
		/// Map x, y from [0 1]^2 square to a semisphere vector3.
		/// If x, y are uniformly distributed, the vector3 will be cosine distributed over the semisphere
		/// </summary>
		/// <returns>The semisphere vector mapped from x, y</returns>
		/// <param name="x">The x coordinate from 0 to 1</param>
		/// <param name="y">The y coordinate from 0 to 1</param>
		public static Vector3 SquareToSemisphereCosineDist(float x, float y){
			x = Mathf.Clamp01 (x);
			y = Mathf.Clamp01 (y);
			float theta1 = Mathf.Acos(Mathf.Sqrt(1 - x));
			float theta2 = 2 * Mathf.PI* y;
			x = Mathf.Cos(theta2)*Mathf.Sin(theta1);
			y = Mathf.Sin(theta2)*Mathf.Cos(theta1);
			float sqrs = x*x + y*y;
			float z= Mathf.Sqrt(1 - sqrs);

			return new Vector3 (x, y, z);
		}
			

		/// <summary>
		/// map [0,1]^2 to unity circle uniformly using Shirley Chiu Mapping
		/// Reference: https://pdfs.semanticscholar.org/4322/6a3916a85025acbb3a58c17f6dc0756b35ac.pdf
		/// </summary>
		/// <returns>The point on unit circle </returns>
		/// <param name="onSquare">Point On square. should be [0,1]^2</param>
		static Vector2 ShirleyChiuMapToDisk(Vector2 onSquare) {
			
			float phi, r, u, v;
			float a = 2 * Mathf.Clamp01 (onSquare.x) - 1; // map from [0 1]^2 to [-1 1]^2
			float b = 2 * Mathf.Clamp01 (onSquare.y) - 1;
			if (a > -b) {
				// region 1 or 2
				if (a > b) {
					// region 1, also |a| > |b|
					r = a;
					phi = (Mathf.PI / 4) * (b / a);
				}else {
					// region 2, also |b| > |a|
					r = b;
					phi = (Mathf.PI / 4) * (2 - (a / b));
				}
			}else {
				// region 3 or 4
				if (a < b) {
					// region 3, also |a| >= |b|, a != 0
					r = -a;
					phi = (Mathf.PI / 4) * (4 + (b / a));
				}
				else {
					// region 4, |b| >= |a|, but a==0 and b==0 could occur.
					r = -b;
					if (b != 0) {
						phi = (Mathf.PI / 4) * (6 - (a / b));
					}
					else {
						phi = 0;
					}
				}
			}				
			u = r* Mathf.Cos(phi);
			v = r* Mathf.Sin(phi);
			return new Vector2(u, v);
		}

		/// <summary>
		/// Forms the basis rotation matrix for the normal
		/// </summary>
		/// <returns>The rotation matrix basis which has n as z</returns>
		/// <param name="n">N. The normal</param>
		public static Matrix4x4 FormBasis(this Vector3 n) {
			//pick a random Q not parallel to n
			Vector3 Q = n.normalized;
			float smallest = Mathf.Min(Mathf.Abs(n.x), Mathf.Abs(n.y), Mathf.Abs(n.z));
			if (smallest == Mathf.Abs(Q.x)) {
				Q.x = 1.0f;
			}
			else if (smallest == Mathf.Abs(Q.y)) {
				Q.y = 1.0f;
			}
			else {
				Q.z = 1.0f;
			}

			Vector3 T = (Vector3.Cross(Q,n)).normalized;
			Vector3 B = (Vector3.Cross(n,T)).normalized;

			Matrix4x4 result = Matrix4x4.identity;
			result.SetColumn(0, T); result.SetColumn(1, B); result.SetColumn(2, n);
			return result;
		}

	}



}






