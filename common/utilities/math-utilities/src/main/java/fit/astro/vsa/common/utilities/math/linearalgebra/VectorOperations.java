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
package fit.astro.vsa.common.utilities.math.linearalgebra;

import fit.astro.vsa.common.bindings.math.vector.ebe.ElementFunction;
import java.util.Arrays;
import org.apache.commons.math3.analysis.function.Exp;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Borrows from
 * <p>
 * Brzustowicz, M. R. (2017). Data Science with Java: Practical Methods for
 * Scientists and Engineers. " O'Reilly Media, Inc.".
 *
 * @author Kyle.Johnston
 */
public class VectorOperations {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(VectorOperations.class);

    // the floating point precision level to use when testing a value for "zero"
    private static final double ZERO_FLOAT_PRECISION = 2 * Double.MIN_VALUE;
    private static final String UNEQUAL = "Unequal Array Sizes";

    //Empty Constructor
    private VectorOperations() {

    }

    // ======================================================================
    // <editor-fold defaultstate="collapsed" desc="On Element Operations">
    public static RealVector ebeOperations(RealVector a, RealVector b,
            ElementFunction function) {

        RealVector c = new ArrayRealVector(a.getDimension());

        if (a.getDimension() == b.getDimension()) {
            for (int i = 0; i < a.getDimension(); i++) {
                c.setEntry(i, function.eval(a.getEntry(i), b.getEntry(i)));
            }
        } else {
            throw new IllegalArgumentException(UNEQUAL);
        }
        return c;

    }

    // </editor-fold>
    // ======================================================================
    // <editor-fold defaultstate="collapsed" desc="Math">
    /**
     * cross3X3 productOfElements of 2 3x1 vectors
     * http://en.wikipedia.org/wiki/Cross_product
     * http://www.encyclopediaofmath.org/index.php/Cross_product
     *
     * @param left a vector of length 3
     * @param right a vector of length 3
     *
     * @return a cross3X3 b
     * @throws IllegalArgumentException if the size of <tt>a != b</tt>
     * and length of <tt> a = 3 </tt>
     */
    public static double[] cross3(double[] left, double[] right) {
        if ((left.length != 3) || (right.length != 3)) {
            throw new IllegalArgumentException(UNEQUAL);
        }

        double[] result = new double[3];
        result[0] = left[1] * right[2] - left[2] * right[1];
        result[1] = left[2] * right[0] - left[0] * right[2];
        result[2] = left[0] * right[1] - left[1] * right[0];

        return result;
    }

    /**
     *
     * @param leftVector
     * @param rightVector
     * @return
     * @throws IllegalArgumentException if the size of <tt>a != b</tt>
     * and length of <tt> a = 3 </tt>
     */
    public static RealVector cross3(RealVector leftVector, RealVector rightVector) {
        if ((leftVector.getDimension() != 3) || (rightVector.getDimension() != 3)) {
            throw new IllegalArgumentException(UNEQUAL);
        }

        return new ArrayRealVector(cross3(leftVector.toArray(), rightVector.toArray()));
    }

    /**
     * Vector Product
     *
     * @param a
     * @return
     */
    public static double productOfElements(RealVector a) {
        double product = 1;

        for (int i = 0; i < a.getDimension(); i++) {
            product *= a.getEntry(i);
        }
        return product;
    }

    /**
     * Array Summation
     *
     * @param a
     * @return
     */
    public static double summationOfElements(RealVector a) {
        double sum = 0;

        for (int i = 0; i < a.getDimension(); i++) {
            sum += a.getEntry(i);
        }
        return sum;
    }

    /**
     * Cumulative Product on elements in ascending index
     *
     * http://www.mathworks.com/help/matlab/ref/cumprod.html?refresh=true
     *
     * @param a
     *
     * @return
     */
    public static RealVector cumulativeProductOfElements(RealVector a) {
        RealVector b = MatrixUtils.createRealVector(a.toArray());

        for (int i = 1; i < a.getDimension(); i++) {
            b.setEntry(i, a.getEntry(i) * b.getEntry(i - 1));
        }

        return b;
    }

    /**
     * Cumulative Sum on elements in ascending index
     *
     * http://www.mathworks.com/help/matlab/ref/cumsum.html
     *
     * @param a
     *
     * @return
     */
    public static RealVector cumulativeSummationOfElements(RealVector a) {
        RealVector b = MatrixUtils.createRealVector(a.toArray());

        for (int i = 1; i < a.getDimension(); i++) {
            b.setEntry(i, a.getEntry(i) + b.getEntry(i - 1));
        }

        return b;
    }

    /**
     * Shift the elements of the array by m, circularly (end to start)
     *
     * http://en.wikipedia.org/wiki/Circular_shift
     *
     * @param a
     * @param m
     *
     * @return
     */
    public static double[] circularShift(double[] a, int m) {
        double[] b = new double[a.length];
        int counter = 0;

        for (int i = a.length - m; i < a.length; i++) {
            b[counter] = a[i];
            counter++;
        }

        System.arraycopy(a, 0, b, m, a.length - m);

        return b;
    }

    // </editor-fold>
    // ======================================================================
    // <editor-fold defaultstate="collapsed" desc="Statistics">
    /**
     * Standardizes array by subtracting sample mean and dividing by sample
     * standard deviation for each element. Efficient calculation for sample
     * 'mean' and 'variance'
     *
     * See mValue. Hoemmen, Computing the standard deviation efficiently,
     * http://www.cs.berkeley.edu/˜mhoemmen/cs194/Tutorials/variance.pdf (2007).
     *
     * and Higham, 2002, Accuracy and Stability of Numerical Algorithms, 2nd ed
     * SIAM
     *
     *
     * @param a
     *
     * @return
     */
    public static RealVector standardize(RealVector a) {

        double mValue = a.getEntry(0);  // (Cumulative) Sample mean
        double qValue = 0d;      // (Cumulative) Sample variance

        int arraySize = a.getDimension();
        for (int k = 1; k < arraySize; k++) {
            qValue += k * (a.getEntry(k) - mValue) * (a.getEntry(k) - mValue) / (k + 1);
            mValue += (a.getEntry(k) - mValue) / (k + 1);
        }

        // Sample standard deviation
        qValue = Math.sqrt(qValue / (arraySize - 1));
        for (int k = 0; k < arraySize; k++) {
            a.setEntry(k, (a.getEntry(k) - mValue) / qValue);
        }
        return a;
    }

    /**
     * Sample Mean
     *
     * Higham, 2002, Accuracy and Stability of Numerical Algorithms, 2nd ed SIAM
     *
     * @param a
     *
     * @return
     */
    public static double mean(RealVector a) {

        double mValue = a.getEntry(0);  // (Cumulative) Sample mean

        int arraySize = a.getDimension();
        for (int k = 1; k < arraySize; k++) {
            mValue += (a.getEntry(k) - mValue) / (k + 1);
        }

        return mValue;
    }

    /**
     * Sample StDev
     *
     * Higham, 2002, Accuracy and Stability of Numerical Algorithms, 2nd ed SIAM
     *
     * @param a
     *
     * @return
     */
    public static double std(RealVector a) {

        double[] mK = new double[a.getDimension()];
        double[] qK = new double[a.getDimension()];

        for (int indexi = 0; indexi < a.getDimension(); indexi++) {

            if (indexi == 0) {
                mK[indexi] = a.getEntry(indexi);
            } else {
                mK[indexi] = mK[indexi - 1]
                        + (a.getEntry(indexi) - mK[indexi - 1]) / (indexi + 1);
            }

            if (indexi == 0) {
                qK[indexi] = 0;
            } else {
                qK[indexi] = qK[indexi - 1]
                        + (indexi) * Math.pow(a.getEntry(indexi) - mK[indexi - 1], 2) / (indexi);
            }

        }

        return Math.sqrt(qK[a.getDimension() - 1] / (a.getDimension() - 1));
    }

    /**
     * Interquartile range
     *
     * @param a
     * @return
     */
    public static double iqr(double[] a) {

        Percentile percentile = new Percentile();
        return percentile.evaluate(a, 75) - percentile.evaluate(a, 25);
    }

    /**
     * Shannon Entropy
     *
     * http://en.wikipedia.org/wiki/Entropy_(information_theory)#Characterization
     *
     * @param x
     *
     * @return
     */
    public static double computeEntropy(double[] x) {

        double entropy = 0;
        for (int indexi = 0; indexi < x.length; indexi++) {
            if (x[indexi] > 0) {
                entropy += x[indexi] * Math.log(x[indexi]);
            }
        }
        return -entropy;
    }

    /**
     * Mimic's the diff function in MATLAB
     *
     * @param a
     *
     * @return
     */
    public static double[] diffArray(double[] a) {
        double[] diff = new double[a.length - 1];

        for (int i = 1; i < a.length; i++) {
            diff[i - 1] = a[i] - a[i - 1];
        }

        return diff;
    }

    /**
     *
     * @param a
     * @return
     */
    public static RealVector diffArray(RealVector a) {
        return new ArrayRealVector(diffArray(a.toArray()));
    }

    /**
     *
     * @param a
     *
     * @return
     */
    public static double median(double[] a) {
        Percentile percentile = new Percentile();
        return percentile.evaluate(a, 50);
    }

    // </editor-fold>
    // ======================================================================
    // <editor-fold defaultstate="collapsed" desc="Generation">
    /**
     *
     * http://www.mathworks.com/help/stats/quantile.html
     *
     * @param a distribution
     * @param evalArray set of pth percentile
     *
     * @return
     */
    public static double[] quantileSpace(double[] a, double[] evalArray) {

        double[] quantileSpaceArray = new double[evalArray.length];
        Percentile percentile = new Percentile();
        for (int indexi = 0; indexi < evalArray.length; indexi++) {
            quantileSpaceArray[indexi]
                    = percentile.evaluate(a, evalArray[indexi]);
        }
        return quantileSpaceArray;
    }

    /**
     * Generate Linearly Space Vector
     *
     * http://www.mathworks.com/help/matlab/ref/linspace.html
     *
     * @param max
     * @param min
     * @param dimensions
     *
     * @return
     */
    public static double[] linearSpace(double max, double min, int dimensions) {

        double[] linearArray = new double[dimensions];

        double binSize = (max - min) / (dimensions - 1);

        for (int indexi = 0; indexi < dimensions; indexi++) {
            linearArray[indexi] = min + binSize * (double) indexi;
        }

        linearArray[dimensions - 1] = max;
        return linearArray;
    }

    /**
     * Generate Linearly Space Vector
     *
     * http://www.mathworks.com/help/matlab/ref/linspace.html
     *
     * @param max
     * @param min
     * @param interval
     *
     * @return
     */
    public static double[] linearSpace(double max, double min, double interval) {

        long dimensions = Math.round((max - min) / interval + 1);

        double[] linearArray = new double[(int) dimensions];

        for (int indexi = 0; indexi < dimensions; indexi++) {
            linearArray[indexi] = min + interval * (double) indexi;
        }

        if (Math.abs(linearArray[(int) dimensions - 1] - max) <= ZERO_FLOAT_PRECISION) {
            linearArray[(int) dimensions - 1] = max;
        }

        return linearArray;
    }
    
        /**
     * Generate Linearly Space Vector
     *
     * http://www.mathworks.com/help/matlab/ref/linspace.html
     *
     * @param maxLin
     * @param minLin
     * @param interval
     *
     * @return
     */
    public static double[] logSpace(double maxLin, double minLin, double interval) {

        double max = Math.log(maxLin);
        double min = Math.log(minLin);
        
        
        long dimensions = Math.round((max - min) / interval + 1);

        double[] linearArray = new double[(int) dimensions];

        for (int indexi = 0; indexi < dimensions; indexi++) {
            linearArray[indexi] = min + interval * (double) indexi;
        }

        if (Math.abs(linearArray[(int) dimensions - 1] - max) <= ZERO_FLOAT_PRECISION) {
            linearArray[(int) dimensions - 1] = max;
        }

        RealVector tmpLin = new ArrayRealVector(linearArray);
        
        return tmpLin.map(new Exp()).toArray();
    }

    /**
     * Generate Linearly Space Vector
     *
     * http://www.mathworks.com/help/matlab/ref/linspace.html
     *
     * @param max
     * @param min
     * @param interval
     *
     * @return
     */
    public static int[] linearSpace(int max, int min, int interval) {

        int dimensions = (max - min) / interval + 1;

        int[] linearArray = new int[dimensions];

        for (int indexi = 0; indexi < dimensions; indexi++) {
            linearArray[indexi] = min + interval * indexi;
        }

        if (linearArray[(int) dimensions - 1] != max) {
            linearArray[(int) dimensions - 1] = max;
        }

        return linearArray;
    }

    /**
     * Creates an array of 1.0
     *
     * http://www.mathworks.com/help/matlab/ref/ones.html?searchHighlight=ones
     *
     * @param size
     *
     * @return
     */
    public static double[] ones(int size) {
        double[] onesArray = new double[size];
        Arrays.fill(onesArray, 1.0);
        return onesArray;
    }

    /**
     * Creates an array of 1
     *
     * @param size
     *
     * @return
     */
    public static int[] onesInt(int size) {
        int[] onesArray = new int[size];
        Arrays.fill(onesArray, 1);
        return onesArray;
    }

    // </editor-fold>
}
