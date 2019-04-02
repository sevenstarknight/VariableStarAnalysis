/*
 * Copyright (C) 2016 Kyle Johnston
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without isEven the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package fit.astro.vsa.common.utilities.math.linearalgebra;

import fit.astro.vsa.common.bindings.math.matrix.UnivariateFunctionMapper;
import java.util.List;
import fit.astro.vsa.common.bindings.math.geometry.Angle;
import fit.astro.vsa.common.utilities.math.NumericTests;
import java.util.Iterator;
import java.util.Map;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.analysis.function.Cos;
import org.apache.commons.math3.analysis.function.Power;
import org.apache.commons.math3.analysis.function.Sin;
import org.apache.commons.math3.analysis.function.Sqrt;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * Various Math functions; many of them for vector and matrix operations for 3
 * dimensions.
 * <p>
 * 6-27-2013: Updated by K.B. Johnston for usage else where 10-1-2013: Big
 * updates to implement the MatrixUtil from apache commons Math this will
 * include using the RealMatrix and RealVector variables
 * <p>
 */
public class MatrixOperations {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(MatrixOperations.class);

    // Empty Constructor
    private MatrixOperations() {

    }

    // ======================================================================
    // <editor-fold defaultstate="collapsed" desc="On Element Operations">
    /**
     * Applies x^2 to Elements
     * <p>
     * @param a
     * <p>
     * @return square(a_i)
     */
    public static RealMatrix elementSquare(RealMatrix a) {

        RealMatrix b = a.copy();
        b.walkInOptimizedOrder(new UnivariateFunctionMapper(new Power(2)));
        return b;
    }

    /**
     * Applies SQRT to Elements
     * <p>
     * @param a
     * <p>
     * @return sqrt(a_i)
     */
    public static RealMatrix elementSqrt(RealMatrix a) {

        RealMatrix b = a.copy();
        b.walkInOptimizedOrder(new UnivariateFunctionMapper(new Sqrt()));
        return b;
    }

    /**
     * Applies Sine to Elements
     * <p>
     * @param a
     * <p>
     * @return sin(a_i)
     */
    public static RealMatrix elementSine(RealMatrix a) {

        RealMatrix b = a.copy();
        b.walkInOptimizedOrder(new UnivariateFunctionMapper(new Sin()));
        return b;
    }

    /**
     * Applies Cosine to Elements
     * <p>
     * @param a
     * <p>
     * @return cosine(a_i)
     */
    public static RealMatrix elementCosine(RealMatrix a) {

        RealMatrix b = a.copy();
        b.walkInOptimizedOrder(new UnivariateFunctionMapper(new Cos()));
        return b;
    }

    /**
     * The EBE Multi or Hadamard Product
     *
     * @param a
     * @param b
     * @return
     */
    public static RealMatrix hadamardProduct(RealMatrix a, RealMatrix b) {

        if (a.getRowDimension() != b.getRowDimension()
                || a.getColumnDimension() != b.getColumnDimension()) {
            throw new IllegalArgumentException("Matrix Sizes Must be Equal");
        }

        RealMatrix hahamard = MatrixUtils.createRealMatrix(a.getRowDimension(),
                a.getColumnDimension());

        for (int idx = 0; idx < a.getRowDimension(); idx++) {
            for (int jdx = 0; jdx < a.getColumnDimension(); jdx++) {
                hahamard.setEntry(idx, jdx,
                        a.getEntry(idx, jdx) * b.getEntry(idx, jdx));
            }
        }

        return hahamard;
    }

    /**
     * The EBE Multi or Hadamard Division, elements of b_ij that are zero result in a zero in the c_ij
     *
     * @param a
     * @param b
     * @return
     */
    public static RealMatrix hadamardDivision(RealMatrix a, RealMatrix b) {

        if (a.getRowDimension() != b.getRowDimension()
                || a.getColumnDimension() != b.getColumnDimension()) {
            throw new IllegalArgumentException("Matrix Sizes Must be Equal");
        }

        RealMatrix hahamard = MatrixUtils.createRealMatrix(a.getRowDimension(),
                a.getColumnDimension());

        for (int idx = 0; idx < a.getRowDimension(); idx++) {
            for (int jdx = 0; jdx < a.getColumnDimension(); jdx++) {

                if (NumericTests.isApproxZero(b.getEntry(idx, jdx))) {
                    hahamard.setEntry(idx, jdx, 0);
                } else {
                    hahamard.setEntry(idx, jdx,
                            a.getEntry(idx, jdx) / b.getEntry(idx, jdx));
                }
            }
        }

        return hahamard;
    }

    // </editor-fold>
    //=================================================================
    // <editor-fold defaultstate="collapsed" desc="Math">
    
    /**
     *
     * @param reducedSet
     * <p>
     * @return
     */
    public static RealMatrix generateMatrixFromList(List<RealVector> reducedSet) {
        int lengthDims = reducedSet.get(0).getDimension();

        RealMatrix reducedSetMatrix = MatrixUtils.createRealMatrix(
                reducedSet.size(), lengthDims);

        int count = 0;
        for (RealVector currentVector : reducedSet) {
            reducedSetMatrix.setRowVector(count, currentVector);
            count++;
        }
        return reducedSetMatrix;
    }

    /**
     * Build Pseudo-inverse (pxL) matrix Gp := inv(G'G)*G' [or V*G']
     * <p>
     * @param G
     * @param V
     * @param L
     * @param p
     * <p>
     * @return
     */
    public static double[][] calcPseudoInverse(double[][] G, double[][] V, int L, int p) {

        return calcPseudoInverse(new Array2DRowRealMatrix(G),
                new Array2DRowRealMatrix(V), L, p).getData();
    }

    /**
     * Build Pseudo-inverse (pxL) matrix Gp := inv(G'G)*G' [or V*G']
     * <p>
     * @param G
     * @param V
     * @param L
     * @param p
     * <p>
     * @return
     */
    public static RealMatrix calcPseudoInverse(RealMatrix G, RealMatrix V, int L, int p) {
        RealMatrix Gp = MatrixUtils.createRealMatrix(p, L);// Pseudo-inverse of (pxL) G
        double sum;

        for (int i = 0; i < p; i++) {
            for (int j = 0; j < L; j++) {
                sum = 0;
                for (int k = 0; k < p; k++) {
                    sum += V.getEntry(i, k) * G.getEntry(j, k);
                }
                Gp.setEntry(i, j, sum);
            }
        }
        return Gp;
    }

    /**
     * Build Symmetric Gramian
     * <p>
     * http://en.wikipedia.org/wiki/Gramian_matrix
     * http://www.encyclopediaofmath.org/index.php/Gram_matrix
     * <p>
     * @param G Gradient/Coefficient (Lxp) matrix
     * @param L dimension
     * @param p dimension
     * <p>
     * @return
     */
    public static double[][] calcSymmetricGrammian(double[][] G, int L, int p) {

        return calcSymmetricGrammian(new Array2DRowRealMatrix(G), L, p).getData();
    }

    /**
     * Build Symmetric Gramian
     * <p>
     * http://en.wikipedia.org/wiki/Gramian_matrix
     * http://www.encyclopediaofmath.org/index.php/Gram_matrix
     * <p>
     * @param G Gradient/Coefficient (Lxp) matrix
     * @param L dimension
     * @param p dimension
     * <p>
     * @return
     */
    public static RealMatrix calcSymmetricGrammian(RealMatrix G, int L, int p) {
        RealMatrix V = MatrixUtils.createRealMatrix(p, p);// Symmetric Grammian (pxp) matrix
        double sum;

        // Build Symmetric Grammian (pxp) matrix V := G'G
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                sum = 0.0;
                for (int k = 0; k < L; k++) {
                    sum += G.getEntry(k, i) * G.getEntry(k, j);
                }
                V.setEntry(i, j, sum);
            }
        }
        return V;
    }

    /**
     * Calculates the inverse of square matrix 'A'
     * <p>
     * http://en.wikipedia.org/wiki/Invertible_matrix
     * http://www.encyclopediaofmath.org/index.php/Inversion_of_a_matrix
     * <p>
     * @param A input square matrix
     * <p>
     * @return
     */
    public static double[][] calcInverse(double[][] A) throws IllegalArgumentException {

        int dim = A[1].length;
        double det, invdet;
        double[][] result = new double[dim][dim];

        switch (dim) {

            /**
             * http://en.wikipedia.org/wiki/Invertible_matrix#Inversion_of_2.C3.972_matrices
             */
            case 2:
                det = A[0][0] * A[1][1] - A[0][1] * A[1][0];

                invdet = 1 / det;
                result[0][0] = A[1][1] * invdet;
                result[0][1] = -A[0][1] * invdet;
                result[1][0] = -A[1][0] * invdet;
                result[1][1] = A[0][0] * invdet;

                break;

            /**
             * http://en.wikipedia.org/wiki/Invertible_matrix#Inversion_of_3.C3.973_matrices
             */
            case 3:

                det = A[0][0] * (A[1][1] * A[2][2] - A[2][1] * A[1][2])
                        - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
                        + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);

                invdet = 1 / det;
                result[0][0] = (A[1][1] * A[2][2] - A[2][1] * A[1][2]) * invdet;
                result[0][1] = -(A[0][1] * A[2][2] - A[0][2] * A[2][1]) * invdet;
                result[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * invdet;
                result[1][0] = -(A[1][0] * A[2][2] - A[1][2] * A[2][0]) * invdet;
                result[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * invdet;
                result[1][2] = -(A[0][0] * A[1][2] - A[1][0] * A[0][2]) * invdet;
                result[2][0] = (A[1][0] * A[2][1] - A[2][0] * A[1][1]) * invdet;
                result[2][1] = -(A[0][0] * A[2][1] - A[2][0] * A[0][1]) * invdet;
                result[2][2] = (A[0][0] * A[1][1] - A[1][0] * A[0][1]) * invdet;

                break;

            default:
                LOGGER.warn("Direct computation of inverse not avalible "
                        + "for dimensions > 3, use SVD");

                // Implementation of RealMatrix using a double[][] array to store entries.
                RealMatrix M = MatrixUtils.createRealMatrix(A);

                SingularValueDecomposition Msvd
                        = new SingularValueDecomposition(M);

                //Returns matrix entries as a two-dimensional array
                result = Msvd.getSolver().getInverse().getData();

                break;

        } // End switch statement

        return result;
    }

    /**
     * is a matrix with the terms of a geometric progression in each row, i.e.,
     * an m × n matrix
     * <p>
     * @param v
     * <p>
     * @return
     */
    public static RealMatrix constructVanderMonde(double[] v) {

        RealMatrix aMatrix = MatrixUtils.createRealMatrix(v.length, v.length);
        aMatrix = aMatrix.scalarAdd(1);

        for (int idx = 0; idx < v.length; idx++) {
            for (int jdx = 0; jdx < v.length; jdx++) {
                aMatrix.setEntry(idx, jdx, Math.pow(v[idx], jdx));
            }
        }

        return aMatrix;
    }

    /**
     * Flips the matrix from left to right
     * <p>
     * @param a
     * <p>
     * @return
     */
    public static RealMatrix flipMatrixLR(RealMatrix a) {

        int columnSize = a.getColumnDimension();
        for (int i = 0; i < columnSize; i++) {

            double[] currentRow = a.getRow(i);
            ArrayUtils.reverse(currentRow);
            a.setRow(i, currentRow);

        }
        return a;
    }

    /**
     * Flips the matrix from Top to Bottom
     * <p>
     * @param a
     * <p>
     * @return
     */
    public static RealMatrix flipMatrixTB(RealMatrix a) {

        int rowSize = a.getRowDimension();
        for (int i = 0; i < rowSize; i++) {

            double[] currentColumn = a.getColumn(i);
            ArrayUtils.reverse(currentColumn);
            a.setColumn(i, currentColumn);

        }
        return a;
    }

    /**
     * Cumulative Product of Matrix in direction either along row or column
     * <p>
     * @param a 2x2 matrix or less
     * @param alongColumn true = along column, false = along row
     * <p>
     * @return
     */
    public static RealMatrix cumulativeDimensionalProd(RealMatrix a, boolean alongColumn) throws IllegalArgumentException {

        RealMatrix b
                = MatrixUtils.createRealMatrix(a.getRowDimension(),
                        a.getColumnDimension());

        if (alongColumn) {
            for (int i = 0; i < a.getColumnDimension(); i++) {
                b.setColumnVector(i, VectorOperations.cumulativeProductOfElements(a.getColumnVector(i)));
            }
        } else {
            for (int i = 0; i < a.getRowDimension(); i++) {
                b.setRowVector(i, VectorOperations.cumulativeProductOfElements(a.getRowVector(i)));
            }
        }

        return b;
    }

    /**
     * Cumulative Summation of Matrix in direction either along row or column
     * <p>
     *
     * @param a 2x2 matrix or less
     * @param alongColumn true = along column, false = along row
     * <p>
     * @return
     */
    public static RealMatrix cumulativeDimensionalSummation(RealMatrix a, boolean alongColumn) throws IllegalArgumentException {

        RealMatrix b
                = MatrixUtils.createRealMatrix(a.getRowDimension(),
                        a.getColumnDimension());

        if (alongColumn) {
            for (int i = 0; i < a.getColumnDimension(); i++) {
                b.setColumnVector(i, VectorOperations.cumulativeSummationOfElements(a.getColumnVector(i)));
            }
        } else {
            for (int i = 0; i < a.getRowDimension(); i++) {
                b.setRowVector(i, VectorOperations.cumulativeSummationOfElements(a.getRowVector(i)));
            }
        }

        return b;
    }

    // </editor-fold>
    // ======================================================================
    // <editor-fold defaultstate="collapsed" desc="Statistics">
    /**
     * Sum along dimensions
     * <p>
     * x_total_i = sum_i(x_i)
     * <p>
     * @param a matrix of (n,m) dimensions
     * @param alongColumn true = along column, false = along row
     * <p>
     * @return
     */
    public static double[] dimensionalSummation(double[][] a, boolean alongColumn) {

        double[] sum;

        if (alongColumn) {
            sum = new double[a.length];
            for (int indexi = 0; indexi < a.length; indexi++) {
                for (int indexj = 0; indexj < a[0].length; indexj++) {
                    sum[indexi] += a[indexi][indexj];
                }
            }
        } else {
            sum = new double[a[0].length];
            for (double[] a1 : a) {
                for (int indexj = 0; indexj < a[0].length; indexj++) {
                    sum[indexj] += a1[indexj];
                }
            }
        }

        return sum;
    }

    /**
     * Sum along dimensions
     * <p>
     * x_total_i = sum_i(x_i)
     * <p>
     * @param a matrix of (n,m) dimensions
     * @param alongColumn true = along column, false = along row
     * <p>
     * @return
     */
    public static RealVector dimensionalSummation(RealMatrix a, boolean alongColumn) {

        double[] sum;

        if (alongColumn) {
            sum = new double[a.getColumnDimension()];
            for (int indexi = 0; indexi < a.getColumnDimension(); indexi++) {
                sum[indexi] = VectorOperations.summationOfElements(a.getColumnVector(indexi));
            }
        } else {
            sum = new double[a.getRowDimension()];
            for (int indexi = 0; indexi < a.getRowDimension(); indexi++) {
                sum[indexi] = VectorOperations.summationOfElements(a.getRowVector(indexi));
            }
        }

        return MatrixUtils.createRealVector(sum);
    }

    /**
     * Mean along dimensions
     * <p>
     * x_total_i = mean_i(x_i)
     * <p>
     * @param a matrix of (n,m) dimensions
     * @param alongColumn true = along column, false = along row
     * <p>
     * @return
     */
    public static RealVector dimensionalMean(RealMatrix a, boolean alongColumn) {

        int rowDim = a.getRowDimension();
        int colDim = a.getColumnDimension();
        double[] mean;

        if (alongColumn) {
            mean = new double[colDim];
            for (int indexi = 0; indexi < colDim; indexi++) {
                mean[indexi] = VectorOperations.summationOfElements(
                        a.getColumnVector(indexi)) / (double) rowDim;
            }
        } else {
            mean = new double[rowDim];
            for (int indexi = 0; indexi < rowDim; indexi++) {
                mean[indexi] = VectorOperations.summationOfElements(
                        a.getRowVector(indexi)) / (double) colDim;
            }
        }

        return MatrixUtils.createRealVector(mean);
    }

    /**
     * Median along dimensions
     * <p>
     * x_total_i = median_i(x_i)
     * <p>
     * @param a matrix of (n,m) dimensions
     * @param alongColumn true = along column, false = along row
     * <p>
     * @return
     */
    public static RealVector dimensionalMedian(RealMatrix a, boolean alongColumn) {

        double[] median;

        if (alongColumn) {
            median = new double[a.getColumnDimension()];
            for (int indexi = 0; indexi < a.getColumnDimension(); indexi++) {
                median[indexi] = VectorOperations.median(a.getColumn(indexi));
            }
        } else {
            median = new double[a.getRowDimension()];
            for (int indexi = 0; indexi < a.getRowDimension(); indexi++) {
                median[indexi] = VectorOperations.median(a.getRow(indexi));
            }
        }

        return MatrixUtils.createRealVector(median);
    }

    /**
     * Estimate Sum of Matrix, i.e x_total = sum_ij(x_ij)
     * <p>
     * @param a matrix of (n,m) dimensions
     * <p>
     * @return
     */
    public static double sumOfElements(double[][] a) {
        double sum = 0.0;
        for (double[] a1 : a) {
            for (int indexj = 0; indexj < a[0].length; indexj++) {
                sum += a1[indexj];
            }
        }
        return sum;
    }

    /**
     * Estimate Mean of Matrix, i.e. x_bar = sum_ij(x_ij)/(n*m)
     * <p>
     * @param a matrix of (n,m) dimensions
     * <p>
     * @return
     */
    public static double meanOfElements(double[][] a) {
        double sum = sumOfElements(a);
        return sum / (a.length * a[0].length);
    }

    /**
     * Estimate Entropy for the Matrix, i.e. -sum_ij(x_ij*ln(x_ij))
     * <p>
     * @param a matrix of (n,m) dimensions
     * <p>
     * @return
     */
    public static double computeEntropy(double[][] a) {

        double entropy = 0;
        for (double[] xArray : a) {
            for (int indexj = 0; indexj < a[0].length; indexj++) {
                if (xArray[indexj] > 0) {
                    entropy += xArray[indexj] * Math.log(xArray[indexj]);
                }
            }
        }
        return -entropy;
    }

    // </editor-fold>
    //=================================================================
    // <editor-fold defaultstate="collapsed" desc="Generation">
    /**
     *
     * @param reducedSet
     * <p>
     * @return
     */
    public static RealMatrix generateMatrixFromMap(Map<Integer, RealVector> reducedSet) {
        Iterator<Integer> iterMap = reducedSet.keySet().iterator();

        RealVector startingVector = reducedSet.get(iterMap.next());
        int lengthDims = startingVector.getDimension();

        RealMatrix reducedSetMatrix = MatrixUtils.createRealMatrix(
                reducedSet.size(), lengthDims);

        reducedSetMatrix.setRowVector(0, startingVector);

        int count = 1;
        while (iterMap.hasNext()) {
            RealVector currentVector = reducedSet.get(iterMap.next());
            reducedSetMatrix.setRowVector(count, currentVector);
            count++;
        }
        return reducedSetMatrix;
    }

    /**
     * Column stacks a matrix to a vector
     * <p>
     * @param a
     * <p>
     * @return
     */
    public static double[] unpackMatrix(double[][] a) {
        int aRows = a.length;
        int aColumns = a[0].length;

        double[] b = new double[aRows * aColumns];
        int counter = 0;
        for (int i = 0; i < aRows; i++) // row
        {
            for (int j = 0; j < aColumns; j++) {
                b[counter] = a[i][j];
                counter += 1;
            }
        }

        return b;
    }

    /**
     *
     * @param matrix
     * <p>
     * @return
     */
    public static double[] unpackMatrix(RealMatrix matrix) {

        double[] outArray = new double[matrix.getColumnDimension() * matrix.getRowDimension()];
        int count = 0;

        for (int n = 0; n < matrix.getColumnDimension(); n++) {
            double[] tmpColumn = matrix.getColumn(n);
            for (int m = 0; m < matrix.getRowDimension(); m++) {
                outArray[count] = tmpColumn[m];
                count += 1;
            }
        }

        return outArray;

    }

    /**
     *
     * Packs a vector into a matrix column wise (with length m)
     * <p>
     * @param vals
     * @param m
     * <p>
     * @return @throws IllegalArgumentException if Array length is not a
     * multiple of m.
     */
    public static double[][] packMatrix(double[] vals, int m) throws IllegalArgumentException {

        int n = (m != 0 ? vals.length / m : 0);
        if (m * n != vals.length) {
            throw new IllegalArgumentException("Array length must be "
                    + "a multiple of m.");
        }
        double[][] A = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = vals[i + j * m];
            }
        }

        return A;
    }

    /**
     * mimics MATLAB's meshGridX function
     * <p>
     * @param a vector to be repeated
     * @param size width of matrix
     * <p>
     * @return
     */
    public static double[][] meshGridX(double[] a, int size) {
        double[][] meshGrid = new double[a.length][size];

        for (int indexj = 0; indexj < a.length; indexj++) {
            for (int indexi = 0; indexi < size; indexi++) {
                meshGrid[indexj][indexi] = a[indexj];
            }
        }
        return meshGrid;
    }

    /**
     * mimics MATLAB's meshGridY function
     * <p>
     * @param a vector to be repeated
     * @param size height of matrix
     * <p>
     * @return
     */
    public static double[][] meshGridY(double[] a, int size) {
        double[][] meshGrid = new double[size][a.length];

        for (int indexi = 0; indexi < size; indexi++) {
            System.arraycopy(a, 0, meshGrid[indexi], 0, a.length);
        }
        return meshGrid;
    }

    /**
     * Construct a 2D histogram based on the matrix xyData and the bin edges
     * (xEdges) and (YEdges)
     * <p>
     * @param xyData double[0][i] = X coord, double[1][i] = Y Coord
     * @param xEdges
     * @param yEdges
     * <p>
     * @return
     */
    public static double[][] twoDHistogram(double[][] xyData, double[] xEdges,
            double[] yEdges) {

        double[][] histogram2D = new double[xEdges.length][yEdges.length];

        for (int indexj = 0; indexj < xEdges.length; indexj++) {
            for (int indexk = 0; indexk < yEdges.length; indexk++) {
                histogram2D[indexj][indexk] = 0.0;
            }
        }

        for (int indexi = 0; indexi < xyData[0].length; indexi++) {

            boolean found = false;

            for (int indexj = 0; indexj < xEdges.length - 1; indexj++) {
                for (int indexk = 0; indexk < yEdges.length - 1; indexk++) {
                    if (xyData[0][indexi] >= xEdges[indexj]
                            && xyData[0][indexi] < xEdges[indexj + 1]) {

                        if (xyData[1][indexi] >= yEdges[indexk]
                                && xyData[1][indexi] < yEdges[indexk + 1]) {
                            histogram2D[indexj][indexk] += 1.0;
                            found = true;
                            break;
                        }
                    }
                }
                if (found) {
                    break;
                }
            }
        }

        return histogram2D;
    }

    /**
     *
     * @param a
     * @param columns
     * <p>
     * @return
     */
    public static RealMatrix replicateMatrixColumns(RealVector a, int columns) {
        RealMatrix c = MatrixUtils.createRealMatrix(a.getDimension(), columns);
        for (int i = 0; i < columns; i++) {
            c.setColumnVector(i, a);
        }

        return c;
    }

    /**
     *
     * @param a
     * @param rows
     * <p>
     * @return
     */
    public static RealMatrix replicateMatrixRows(RealVector a, int rows) {
        RealMatrix c = MatrixUtils.createRealMatrix(rows, a.getDimension());
        for (int i = 0; i < rows; i++) {
            c.setRowVector(i, a);
        }
        return c;
    }

    /**
     *
     * @param a
     * <p>
     * @return
     */
    public static RealVector getDiag(RealMatrix a) {
        RealVector diag = new ArrayRealVector(a.getRowDimension());
        for (int i = 0; i < a.getRowDimension(); i++) {
            diag.setEntry(i, a.getEntry(i, i));
        }
        return diag;
    }

    // </editor-fold>
    // ======================================================================
    // <editor-fold defaultstate="collapsed" desc="Rotation Matrix">
    /**
     * Elementary rotation matrix about x axis
     * <p>
     * Diebel, J., (2006), "Representing Attitude: Euler Angles, Unit
     * Quaternions, and Rotation Vectors", Standford University
     * <p>
     * @param angle Angle in radians
     * <p>
     * @return Elementary rotation matrix about x axis
     */
    public static double[][] Rx(Angle angle) {
        final double C = angle.cos();
        final double S = angle.sin();
        double[][] U = new double[3][3];
        U[0][0] = 1.0;
        U[0][1] = 0.0;
        U[0][2] = 0.0;
        U[1][0] = 0.0;
        U[1][1] = +C;
        U[1][2] = +S;
        U[2][0] = 0.0;
        U[2][1] = -S;
        U[2][2] = +C;
        return U;
    }

    /**
     * Elementary rotation matrix about y axis
     * <p>
     * Diebel, J., (2006), "Representing Attitude: Euler Angles, Unit
     * Quaternions, and Rotation Vectors", Standford University
     * <p>
     * @param angle angle in radians
     * <p>
     * @return Elementary rotation matrix about y axis
     */
    public static double[][] Ry(Angle angle) {
        final double C = angle.cos();
        final double S = angle.sin();
        double[][] U = new double[3][3];
        U[0][0] = +C;
        U[0][1] = 0.0;
        U[0][2] = -S;
        U[1][0] = 0.0;
        U[1][1] = 1.0;
        U[1][2] = 0.0;
        U[2][0] = +S;
        U[2][1] = 0.0;
        U[2][2] = +C;
        return U;
    }

    /**
     * Elementary rotation matrix about z axis
     * <p>
     * Diebel, J., (2006), "Representing Attitude: Euler Angles, Unit
     * Quaternions, and Rotation Vectors", Standford University
     * <p>
     * @param angle Angle in radians
     * <p>
     * @return Elementary rotation matrix about z axis
     */
    public static double[][] Rz(Angle angle) {
        final double C = angle.cos();
        final double S = angle.sin();
        double[][] U = new double[3][3];
        U[0][0] = +C;
        U[0][1] = +S;
        U[0][2] = 0.0;
        U[1][0] = -S;
        U[1][1] = +C;
        U[1][2] = 0.0;
        U[2][0] = 0.0;
        U[2][1] = 0.0;
        U[2][2] = 1.0;
        return U;
    }

    /**
     * Diebel, J., (2006), "Representing Attitude: Euler Angles, Unit
     * Quaternions, and Rotation Vectors", Standford University
     * <p>
     * @param angle Angle in radians
     * <p>
     * @return
     */
    public static RealMatrix RxMatrix(Angle angle) {
        return MatrixUtils.createRealMatrix(Rx(angle));
    }

    /**
     * Diebel, J., (2006), "Representing Attitude: Euler Angles, Unit
     * Quaternions, and Rotation Vectors", Standford University
     * <p>
     * @param angle Angle in radians
     * <p>
     * @return
     */
    public static RealMatrix RyMatrix(Angle angle) {
        return MatrixUtils.createRealMatrix(Ry(angle));
    }

    /**
     * Diebel, J., (2006), "Representing Attitude: Euler Angles, Unit
     * Quaternions, and Rotation Vectors", Standford University
     * <p>
     * @param angle Angle in radians
     * <p>
     * @return
     */
    public static RealMatrix RzMatrix(Angle angle) {
        return MatrixUtils.createRealMatrix(Rz(angle));
    }

    // </editor-fold>
}
