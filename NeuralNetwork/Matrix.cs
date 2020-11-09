using System;
using System.Diagnostics.Contracts;
using System.Threading.Tasks;

namespace NeuralNetwork
{
	public class Matrix
	{
		private readonly float[,] values;

		public int RowCount => values.GetLength(0);
		public int ColumnCount => values.GetLength(1);

		public Matrix(int rowCount, int columnCount)
		{
			values = new float[rowCount, columnCount];
		}

		public float this[int x, int y]
		{
			get => values[x, y];
			set => values[x, y] = value;
		}

		public static Matrix operator +(Matrix a, Matrix b)
		{
			Contract.Requires(a.RowCount == b.RowCount);
			Contract.Requires(a.ColumnCount == b.ColumnCount);

			var result = new Matrix(a.RowCount, a.ColumnCount);

			Parallel.For(0, a.RowCount, x =>
			{
				for(int y = 0; y < a.ColumnCount; ++y)
				{
					result[x, y] = a[x, y] + b[x, y];
				}
			});

			return result;
		}

		public static Matrix operator +(Matrix matrix, float n)
		{
			var result = new Matrix(matrix.RowCount, matrix.ColumnCount);

			Parallel.For(0, matrix.RowCount, x =>
			{
				for(int y = 0; y < matrix.ColumnCount; ++y)
				{
					result[x, y] = matrix[x, y] + n;
				}
			});

			return result;
		}

		public static Matrix operator +(float n, Matrix matrix)
		{
			return matrix + n;
		}

		public static Matrix operator -(Matrix matrix, float n)
		{
			var result = new Matrix(matrix.RowCount, matrix.ColumnCount);

			Parallel.For(0, matrix.RowCount, x =>
			{
				for(int y = 0; y < matrix.ColumnCount; ++y)
				{
					result[x, y] = matrix[x, y] - n;
				}
			});

			return result;
		}

		public static Matrix operator -(float n, Matrix matrix)
		{
			var result = new Matrix(matrix.RowCount, matrix.ColumnCount);

			Parallel.For(0, matrix.RowCount, x =>
			{
				for(int y = 0; y < matrix.ColumnCount; ++y)
				{
					result[x, y] = n - matrix[x, y];
				}
			});

			return result;
		}

		public static Matrix operator -(Matrix a, Matrix b)
		{
			Contract.Requires(a.RowCount == b.RowCount);
			Contract.Requires(a.ColumnCount == b.ColumnCount);

			var result = new Matrix(a.RowCount, a.ColumnCount);

			Parallel.For(0, a.RowCount, x =>
			{
				for(int y = 0; y < a.ColumnCount; ++y)
				{
					result[x, y] = a[x, y] - b[x, y];
				}
			});

			return result;
		}

		public static Matrix operator *(Matrix a, Matrix b)
		{
			Contract.Requires(a.ColumnCount == b.RowCount);

			var result = new Matrix(a.RowCount, b.ColumnCount);

			Parallel.For(0, result.RowCount, x =>
			{
				for(int y = 0; y < result.ColumnCount; ++y)
				{
					float sum = 0;

					for(int k = 0; k < a.ColumnCount; ++k)
					{
						sum += a[x, k] * b[k, y];
					}

					result[x, y] = sum;
				}
			});

			return result;
		}

		public static Matrix operator *(Matrix matrix, float multiplier)
		{
			var result = new Matrix(matrix.RowCount, matrix.ColumnCount);

			Parallel.For(0, result.RowCount, x =>
			{
				for(int y = 0; y < result.ColumnCount; ++y)
				{
					result[x, y] = matrix[x, y] * multiplier;
				}
			});

			return result;
		}

		public static Matrix operator *(float multiplier, Matrix matrix)
		{
			return matrix * multiplier;
		}

		public static Matrix Transpose(Matrix matrix)
		{
			var result = new Matrix(matrix.ColumnCount, matrix.RowCount);

			Parallel.For(0, result.RowCount, x =>
			{
				for(int y = 0; y < result.ColumnCount; ++y)
				{
					result[x, y] = matrix[y, x];
				}
			});

			return result;
		}

		public static Matrix ConvertAsRow(float[] values)
		{
			var matrix = new Matrix(1, values.Length);

			for(int i = 0; i < values.Length; ++i)
			{
				matrix[0, i] = values[i];
			}

			return matrix;
		}

		public static Matrix ConvertAsColumn(float[] values)
		{
			var matrix = new Matrix(values.Length, 1);

			for(int i = 0; i < values.Length; ++i)
			{
				matrix[i, 0] = values[i];
			}

			return matrix;
		}

		public static float[] Flatten(Matrix matrix)
		{
			float[] values = new float[matrix.RowCount * matrix.ColumnCount];

			for(int x = 0; x < matrix.RowCount; ++x)
			{
				for(int y = 0; y < matrix.ColumnCount; ++y)
				{
					values[x * matrix.ColumnCount + y] = matrix[x, y];
				}
			}
			return values;
		}
	}
}
