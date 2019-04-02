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
import org.apache.commons.math3.linear.RealVector;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class MatrixOperationsStatisticsTest {

    public MatrixOperationsStatisticsTest() {
    }

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

    @Test
    public void testDiag() {
        RealVector diag = MatrixOperations
                .getDiag(matrixB);

        RealVector diagVector = MatrixUtils
                .createRealVector(new double[]{2, 1, 1});
        double deltaA = diagVector.getDistance(diag);

        assertEquals(0.0, deltaA, EPS);
    }

    @Test
    public void testTotalEntropy() {
        double entropy = MatrixOperations
                .computeEntropy(matrixB.getData());

        assertEquals(-31.911872372514196, entropy, EPS);
    }

    @Test
    public void testTotalMean() {
        double mean = MatrixOperations
                .meanOfElements(matrixB.getData());

        assertEquals(23.0 / 9.0, mean, EPS);
    }

    @Test
    public void testTotalSummation() {
        double total = MatrixOperations
                .sumOfElements(matrixB.getData());

        assertEquals(23.0, total, EPS);
    }

    @Test
    public void testDimensionalMedian() {
        RealVector vectorCol
                = MatrixOperations.dimensionalMedian(matrixB, true);

        RealVector vectorA = MatrixUtils.createRealVector(new double[]{
            6.0, 1.0, 1.0
        });

        RealVector vectorRow
                = MatrixOperations.dimensionalMedian(matrixB, false);

        RealVector vectorB = MatrixUtils.createRealVector(new double[]{
            1.0, 1.0, 1.0
        });

        double deltaA = vectorCol.getDistance(vectorA);
        double deltaB = vectorRow.getDistance(vectorB);

        assertEquals(0.0, deltaA, EPS);
        assertEquals(0.0, deltaB, EPS);
    }

    @Test
    public void testDimensionalSummation() {
        RealVector vectorCol
                = MatrixOperations.dimensionalSummation(matrixB, true);

        RealVector vectorA = MatrixUtils.createRealVector(new double[]{
            17.0, 3.0, 3.0
        });

        RealVector vectorRow
                = MatrixOperations.dimensionalSummation(matrixB, false);

        RealVector vectorB = MatrixUtils.createRealVector(new double[]{
            4.0, 8.0, 11.0
        });

        double deltaA = vectorCol.getDistance(vectorA);
        double deltaB = vectorRow.getDistance(vectorB);

        assertEquals(0.0, deltaA, EPS);
        assertEquals(0.0, deltaB, EPS);
    }
}
