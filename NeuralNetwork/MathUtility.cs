using System;

namespace NeuralNetwork
{
	public static class MathUtility
	{
		public static double Sigmoid(double x)
		{
			return 1 / (1 + Math.Pow(Math.E, -x));
		}

		public static float Sigmoid(float x)
		{
			return 1 / (1 + MathF.Pow(MathF.E, -x));
		}

		public static Matrix Sigmoid(Matrix matrix)
		{
			var result = new Matrix(matrix.RowCount, matrix.ColumnCount);

			for(int x = 0; x < result.RowCount; ++x)
			{
				for(int y = 0; y < result.ColumnCount; ++y)
				{
					result[x, y] = Sigmoid(matrix[x, y]);
				}
			}

			return result;
		}
	}
}
