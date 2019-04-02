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

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.junit.Test;
import static org.junit.Assert.*;
import org.junit.Before;

/**
 *
 * @author Kyle.Johnston
 */
public class MatrixOperationsEBETest {

    private static final double EPS = 1.0e-5;

    private static final double[][] INMATRIXB = new double[][]{
        {2.0, 1.0, 1.0},
        {6.0, 1.0, 1.0},
        {9.0, 1.0, 1.0}};

    private RealMatrix matrixB;

    @Before
    public void setUp() {
        this.matrixB = MatrixUtils.createRealMatrix(INMATRIXB);
    }

    /**
     *
     */
    public MatrixOperationsEBETest() {
    }

    // ===============================================
    @Test
    public void testElementCosine() {

        RealMatrix cos = MatrixOperations.
                elementCosine(matrixB);

        double[][] inMatrixX = new double[][]{
            {Math.cos(2), Math.cos(1), Math.cos(1)},
            {Math.cos(6), Math.cos(1), Math.cos(1)},
            {Math.cos(9), Math.cos(1), Math.cos(1)}};

        RealMatrix delta = MatrixUtils.createRealMatrix(inMatrixX)
                .subtract(cos);

        assertEquals(0.0, delta.getFrobeniusNorm(), EPS);
    }

    @Test
    public void testElementSine() {

        RealMatrix sin = MatrixOperations.
                elementSine(matrixB);

        double[][] inMatrixX = new double[][]{
            {Math.sin(2), Math.sin(1), Math.sin(1)},
            {Math.sin(6), Math.sin(1), Math.sin(1)},
            {Math.sin(9), Math.sin(1), Math.sin(1)}};

        RealMatrix delta = MatrixUtils.createRealMatrix(inMatrixX)
                .subtract(sin);

        assertEquals(0.0, delta.getFrobeniusNorm(), EPS);
    }

    @Test
    public void testElementSqrt() {

        RealMatrix sqrt = MatrixOperations.
                elementSqrt(matrixB);

        double[][] inMatrixX = new double[][]{
            {Math.sqrt(2), 1.0, 1.0},
            {Math.sqrt(6), 1.0, 1.0},
            {3, 1.0, 1.0}};

        RealMatrix delta = MatrixUtils.createRealMatrix(inMatrixX)
                .subtract(sqrt);

        assertEquals(0.0, delta.getFrobeniusNorm(), EPS);
    }

    @Test
    public void testElementSquare() {

        RealMatrix square = MatrixOperations.
                elementSquare(matrixB);

        double[][] inMatrixX = new double[][]{
            {4.0, 1.0, 1.0},
            {36.0, 1.0, 1.0},
            {81.0, 1.0, 1.0}};

        RealMatrix delta = MatrixUtils.createRealMatrix(inMatrixX)
                .subtract(square);

        assertEquals(0.0, delta.getFrobeniusNorm(), EPS);
    }

}
