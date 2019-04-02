/*
 * Copyright (C) 2016 Kyle Johnston
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package fit.astro.vsa.common.utilities.math.solvers;

import fit.astro.vsa.common.bindings.math.Prediction;
import fit.astro.vsa.common.utilities.math.linearalgebra.VectorOperations;
import fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException;
import org.apache.commons.math3.distribution.TDistribution;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularValueDecomposition;

/**
 * Original by Walt Fair Jr. (coefficients++) Translated to Java by Sindharta
 * Tanuwijaya (in 2009)
 * <p>
 * http://pubs.usgs.gov/tm/tm4a8/pdf/TM4-A8.pdf
 * <p>
 * Some updating, editing and formatting has been done to the code in general
 * mostly for appearance, containment, and flow. All of the original mathematics
 * has been retained.
 * <p>
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class WeightedMultipleRegression {

    private final int n;                    // m = Number of data points
    private final int p;                    // p = Number of linear terms

    private RealVector coefficients;        // Coefficients

    private final RealVector yArray;
    private final RealMatrix xArray;
    private final RealMatrix weights;

    private final double alpha = 0.05;

    //===================================
    private double mse;
    private RealVector varianceCoefficients;
    private RealMatrix xPrimeWXInverse;

    /**
     * Allows for yArray values with different errors (weights). w_i =
     * 1/(sigma_i)^2;
     *
     * @param yArray
     * @param xArray
     * @param weights
     * @throws
     * fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException
     */
    public WeightedMultipleRegression(double[] yArray,
            double[][] xArray, double[] weights) throws NotEnoughDataException {
        this.xArray = MatrixUtils.createRealMatrix(xArray);
        this.yArray = MatrixUtils.createRealVector(yArray);
        this.weights = MatrixUtils.createRealDiagonalMatrix(weights);

        this.n = this.yArray.getDimension(); // n = Number of data points

        this.p = this.xArray.getColumnDimension();     // p = Number of linear terms

        regress();

    }

    /**
     * Allows for yArray values with different errors (weights). w_i =
     * 1/(sigma_i)^2;
     * <p>
     * @param yArray
     * @param xArray length = dimensions, width = observations
     * @param weights
     * @throws
     * fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException
     */
    public WeightedMultipleRegression(RealVector yArray,
            RealMatrix xArray, RealMatrix weights) throws NotEnoughDataException {
        this.xArray = xArray;
        this.yArray = yArray;
        this.weights = weights;

        this.n = yArray.getDimension(); // n = Number of data points
        this.p = xArray.getColumnDimension();     // p = Number of linear terms

        regress();

    }

    /**
     * Allows for yArray values with different errors (weights). w_i =
     * 1/(sigma_i)^2;
     * <p>
     * @param yArray
     * @param xArray length = dimensions, width = observations
     * @param weights
     *
     * @throws
     * fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException
     */
    public WeightedMultipleRegression(RealVector yArray,
            RealMatrix xArray, double[] weights) throws NotEnoughDataException {
        this.xArray = xArray;
        this.yArray = yArray;
        this.weights = MatrixUtils.createRealDiagonalMatrix(weights);

        this.n = yArray.getDimension(); // m = Number of data points
        this.p = xArray.getColumnDimension();     // p = Number of linear terms

        regress();

    }

    /**
     * Assumes equally weighted yArray values w = 1;
     * <p>
     * @param yArray
     * @param xArray
     *
     * @throws
     * fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException
     */
    public WeightedMultipleRegression(RealVector yArray,
            RealMatrix xArray) throws NotEnoughDataException {
        this.xArray = xArray;
        this.yArray = yArray;

        // Equal Weight
        this.weights = MatrixUtils.createRealDiagonalMatrix(VectorOperations.ones(yArray.getDimension()));

        this.n = yArray.getDimension(); // m = Number of data points
        this.p = xArray.getColumnDimension();     // p = Number of linear terms

        regress();

    }

    /**
     * Perform the regression
     * <p>
     * @throws java.lang.Exception
     */
    private void regress() throws NotEnoughDataException {

        // If not enough data, don't attempt regression
        if ((n - p) < 1) {
            throw new NotEnoughDataException("not enough data, don't attempt regression");
        }

        // X'*W
        RealMatrix xPrimeW = xArray.transpose().multiply(weights);

        // X'*W*X
        RealMatrix xPrimeWX = xPrimeW.multiply(xArray);

        SingularValueDecomposition Msvd
                = new SingularValueDecomposition(xPrimeWX);

        //Returns matrix entries as a two-dimensional array
        xPrimeWXInverse = Msvd.getSolver().getInverse();

        // X'*W*Y
        RealVector xPrimeWY = xPrimeW.operate(yArray);

        coefficients = xPrimeWXInverse.operate(xPrimeWY);

        performanceMetrics();

    }

    private void performanceMetrics() {

        // Generate SSE
        double sse = 0;
        for (int index = 0; index < n; index++) {
            double delta = yArray.getEntry(index)
                    - coefficients.dotProduct(xArray.getRowVector(index));
            sse = sse + delta * weights.getEntry(index, index);
        }

        mse = sse / (n - p);

        // Generate Variance Coefficients
        double[] varCoefficientsArray = new double[p];
        TDistribution tDistribution = new TDistribution(n - p);
        for (int i = 0; i < p; i++) {
            varCoefficientsArray[i] = tDistribution.density(alpha / 2)
                    * Math.sqrt(mse * xPrimeWXInverse.getEntry(i, i));
        }
        varianceCoefficients = MatrixUtils.createRealVector(varCoefficientsArray);

    }

    /**
     *
     * @param xArray
     *
     * @return
     */
    public Prediction predictNewObservation(RealVector xArray) {

        if (xArray.getDimension() == p) {
            throw new ArithmeticException("Input observation size,"
                    + "must match trained dataset size");
        }

        double yEst = xArray.dotProduct(coefficients);

        TDistribution tDistribution = new TDistribution(n - p);
        double tval = tDistribution.density(alpha / 2);

        double xNew = xArray.dotProduct(xPrimeWXInverse.operate(xArray));

        double var = tval * Math.sqrt(mse * (1 + xNew));

        return new Prediction(yEst, var);
    }

    /**
     * @return the coefficients
     */
    public RealVector getCoefficients() {
        return coefficients;
    }

    /**
     * @return the mse
     */
    public double getMse() {
        return mse;
    }

    /**
     * @return the varianceCoefficients
     */
    public RealVector getVarianceCoefficients() {
        return varianceCoefficients;
    }

}
