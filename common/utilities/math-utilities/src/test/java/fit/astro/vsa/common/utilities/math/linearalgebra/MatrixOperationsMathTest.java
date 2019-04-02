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

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.junit.Test;
import static org.junit.Assert.*;
import org.junit.Before;

/**
 *
 * @author Kyle.Johnston
 */
public class MatrixOperationsMathTest {

    private static final double EPS = 1.0e-5;
    private static final double[] IN_ARRAY_A = new double[]{
        4.0, 2.0, 3.0};

    private static final double[][] INMATRIXB = new double[][]{
        {2.0, 1.0, 1.0},
        {6.0, 1.0, 1.0},
        {9.0, 1.0, 1.0}};

    private static final double[][] INMATRIXC = new double[][]{
        {12.0, 6.0},
        {6.0, 8.0}};

    private RealMatrix matrixB;
    private RealMatrix matrixC;

    @Before
    public void setUp() {
        this.matrixC = MatrixUtils.createRealMatrix(INMATRIXC);
        this.matrixB = MatrixUtils.createRealMatrix(INMATRIXB);

    }

    /**
     *
     */
    public MatrixOperationsMathTest() {
    }

    // ===============================================
    @Test
    public void testCumulativeProduct() {
        RealMatrix cumProd1 = MatrixOperations
                .cumulativeDimensionalProd(matrixB, true);

        double[][] inMatrixX1 = new double[][]{
            {2.0, 1.0, 1.0},
            {12.0, 1.0, 1.0},
            {108.0, 1.0, 1.0}};

        RealMatrix delta1 = MatrixUtils.createRealMatrix(
                inMatrixX1).subtract(cumProd1);

        assertEquals(0.0, delta1.getFrobeniusNorm(), EPS);

        // ===============================================
        RealMatrix cumProd2 = MatrixOperations
                .cumulativeDimensionalProd(matrixB, false);

        double[][] inMatrixX2 = new double[][]{
            {2.0, 2.0, 2.0},
            {6.0, 6.0, 6.0},
            {9.0, 9.0, 9.0}};

        RealMatrix delta2 = MatrixUtils.createRealMatrix(
                inMatrixX2).subtract(cumProd2);

        assertEquals(0.0, delta2.getFrobeniusNorm(), EPS);
    }

    @Test
    public void testCumulativeSummation() {
        RealMatrix cumSum1 = MatrixOperations
                .cumulativeDimensionalSummation(matrixB, true);

        double[][] inMatrixX1 = new double[][]{
            {2.0, 1.0, 1.0},
            {8.0, 2.0, 2.0},
            {17.0, 3.0, 3.0}};

        RealMatrix delta1 = MatrixUtils.createRealMatrix(
                inMatrixX1).subtract(cumSum1);

        assertEquals(0.0, delta1.getFrobeniusNorm(), EPS);

        // ===============================================
        RealMatrix cumSum2 = MatrixOperations
                .cumulativeDimensionalSummation(matrixB, false);

        double[][] inMatrixX2 = new double[][]{
            {2.0, 3.0, 4.0},
            {6.0, 7.0, 8.0},
            {9.0, 10.0, 11.0}};

        RealMatrix delta2 = MatrixUtils.createRealMatrix(
                inMatrixX2).subtract(cumSum2);

        assertEquals(0.0, delta2.getFrobeniusNorm(), EPS);
    }

    @Test
    public void testVanderMonde() {

        RealMatrix vanderMond = MatrixOperations
                .constructVanderMonde(IN_ARRAY_A);

        double[][] inMatrixX = new double[][]{
            {1.0, 4.0, 16.0},
            {1.0, 2.0, 4.0},
            {1.0, 3.0, 9.0}};

        RealMatrix delta = MatrixUtils.createRealMatrix(
                inMatrixX).subtract(vanderMond);

        assertEquals(0.0, delta.getFrobeniusNorm(), EPS);
    }

    @Test
    public void testCalcInverse() {

        RealMatrix inverse = new Array2DRowRealMatrix(MatrixOperations
                .calcInverse(matrixC.getData()));

        double[][] inMatrixX = new double[][]{
            {4, -3},
            {-3, 6}};
        RealMatrix matrixX = MatrixUtils
                .createRealMatrix(inMatrixX);

        matrixX = matrixX.scalarMultiply(1.0 / 30.0);

        RealMatrix delta = matrixX.subtract(inverse);

        assertEquals(0.0, delta.getFrobeniusNorm(), EPS);

        // =============================================
        double[][] inverseArray = MatrixOperations
                .calcInverse(INMATRIXC);

        RealMatrix deltaArray = MatrixUtils.createRealMatrix(
                inverseArray)
                .subtract(matrixX);

        assertEquals(0.0, deltaArray.getFrobeniusNorm(), EPS);
    }

}
