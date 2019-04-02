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

import fit.astro.vsa.common.bindings.math.Real2DCurve;
import static fit.astro.vsa.common.utilities.math.linearalgebra.VectorOperations.*;
import fit.astro.vsa.common.bindings.math.vector.ebe.Atan2ElementFunction;
import fit.astro.vsa.common.bindings.math.vector.ebe.MaxElementFunction;
import fit.astro.vsa.common.bindings.math.vector.ebe.MinElementFunction;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Kyle.Johnston
 */
public class VectorUtilsTest {

    private static final double[] IN_ARRAY_A = new double[]{4.0, 2.0, 3.0};
    private static final double[] IN_ARRAY_B = new double[]{3.0, -4.0, -1.0};
    private static final double[] IN_ARRAY_C = new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    private static final RealVector IN_VECTOR_A
            = MatrixUtils.createRealVector(new double[]{4.0, 2.0, 3.0});

    private static final RealVector IN_VECTOR_B
            = MatrixUtils.createRealVector(new double[]{3.0, -4.0, -1.0});

    private static final Real2DCurve IN_SERIES = new Real2DCurve(IN_ARRAY_C, IN_ARRAY_C);

    private static final double SCALAR_VALUE = 2.0;

    private static final double EPS = 1.0e-5;

    //===================================================================
    // <editor-fold defaultstate="collapsed" desc="Vector Element By Element">
    /**
     *
     */
    @Test
    public void elementATan2Test() {
        RealVector outArray = ebeOperations(IN_VECTOR_A, IN_VECTOR_B, new Atan2ElementFunction());
        RealVector expectedArray = MatrixUtils.createRealVector(new double[]{
            Math.atan2(4.0, 3.0),
            Math.atan2(2.0, -4.0),
            Math.atan2(3.0, -1.0)});

        double delta = outArray.getDistance(expectedArray);

        assertEquals(0.0, delta, EPS);
    }

    /**
     *
     */
    @Test
    public void elementMaxTest() {
        RealVector outArray = ebeOperations(IN_VECTOR_A, IN_VECTOR_B, new MaxElementFunction());
        double[] expectedArray = new double[]{4.0, 2.0, 3.0};
        assertArrayEquals(expectedArray, outArray.toArray(), EPS);
    }

    /**
     *
     */
    @Test
    public void elementMinTest() {
        RealVector outArray = ebeOperations(IN_VECTOR_A, IN_VECTOR_B, new MinElementFunction());
        double[] expectedArray = new double[]{3.0, -4.0, -1.0};
        assertArrayEquals(expectedArray, outArray.toArray(), EPS);
    }
    // </editor-fold>

    //===================================================================
    // <editor-fold defaultstate="collapsed" desc="Vector Math">


    /**
     *
     */
    @Test
    public void cross3Test() {
        double[] out = cross3(IN_ARRAY_A, IN_ARRAY_B);
        double[] expected = new double[]{10, 13, -22};
        assertArrayEquals(expected, out, EPS);
    }


    /**
     *
     */
    @Test
    public void cumsumTest() {
        RealVector outArray = cumulativeSummationOfElements(IN_VECTOR_A);
        double[] expectedArray = new double[]{4.0, 6.0, 9.0};
        assertArrayEquals(expectedArray, outArray.toArray(), EPS);
    }

    /**
     *
     */
    @Test
    public void cumprodTest() {
        RealVector outArray = cumulativeProductOfElements(IN_VECTOR_A);
        double[] expectedArray = new double[]{4.0, 8.0, 24.0};
        assertArrayEquals(expectedArray, outArray.toArray(), EPS);
    }

    /**
     *
     */
    @Test
    public void circularShiftTest() {
        double[] outArray = circularShift(IN_ARRAY_C, 2);
        double[] expectedArray = new double[]{5.0, 6.0, 1.0, 2.0, 3.0, 4.0};
        assertArrayEquals(expectedArray, outArray, EPS);
    }

    // </editor-fold>
    //===================================================================
    // <editor-fold defaultstate="collapsed" desc="Vector Scalar Math">
  
    // </editor-fold>

    // ======================================================================
    // <editor-fold defaultstate="collapsed" desc="Vector Statistics">
    /**
     *
     */
    @Test
    public void standardizeTest() {
        RealVector out = standardize(IN_VECTOR_A);
        double[] expected = new double[]{
            1.0, -1.0, 0.0};
        assertArrayEquals(expected, out.toArray(), EPS);
    }

    /**
     *
     */
    @Test
    public void stdTest() {
        double out = std(IN_VECTOR_A);
        double expected = Math.sqrt(2.0);
        assertEquals(expected, out, EPS);
    }

    /**
     *
     */
    @Test
    public void entropyTest() {
        double out = computeEntropy(IN_ARRAY_A);
        double expected = -10.2273086716;
        assertEquals(expected, out, EPS);
    }

    @Test
    public void diffArrayTest() {
        double[] out = diffArray(IN_ARRAY_A);
        double[] expected = new double[]{-2, 1};
        assertArrayEquals(expected, out, EPS);
    }

    // </editor-fold>
    
    // ======================================================================
    //<editor-fold defaultstate="collapsed" desc="Vector Linear Space">
    /**
     * Test creating a vector that has a specified number of dimensions
     * (divisions).
     */
    @Test
    public void linearSpaceTest_dimension() {
        double[] points = linearSpace(5d, -5d, 7);
        double[] expected = new double[]{-5.0000, -3.3333, -1.6667, 0, 1.6667, 3.3333, 5.0000};
        for (int i = 0; i < points.length; i++) {
            assertEquals(expected[i], points[i], 0.0001);
        }
    }

    /**
     * Test creating a vector that has a specified interval (space) between each
     * point.
     */
    @Test
    public void linearSpaceTest_interval() {
        double[] points = linearSpace(5d, -5d, 1.66667);
        double[] expected = new double[]{-5.0000, -3.3333, -1.6667, 0, 1.6667, 3.3333, 5.0000};
        for (int i = 0; i < points.length; i++) {
            assertEquals(expected[i], points[i], 0.0001);
        }
    }

    /**
     * Test creating a vector that has a specified interval (space) between each
     * point using integers.
     */
    @Test
    public void linearSpaceTest_integers() {
        int[] points = linearSpace(5, -5, 2);
        int[] expected = new int[]{-5, -3, -1, 1, 3, 5};
        for (int i = 0; i < points.length; i++) {
            assertEquals(expected[i], points[i], 0);
        }
    }
    //</editor-fold>

    // ======================================================================
    // <editor-fold defaultstate="collapsed" desc="Vector Generation">

    /**
     *
     */
    @Test
    public void onesTest() {
        double[] out = ones(4);
        double[] expected = new double[]{
            1.0, 1.0, 1.0, 1.0};
        assertArrayEquals(expected, out, EPS);
    }

    /**
     *
     */
    @Test
    public void onesIntTest() {
        int[] out = onesInt(4);
        int[] expected = new int[]{
            1, 1, 1, 1};
        assertArrayEquals(expected, out);
    }

    // </editor-fold>
}
