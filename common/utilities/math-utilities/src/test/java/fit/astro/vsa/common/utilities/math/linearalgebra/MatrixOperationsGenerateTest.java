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

import java.util.ArrayList;
import java.util.List;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;
import static fit.astro.vsa.common.utilities.math.linearalgebra.MatrixOperations.unpackMatrix;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class MatrixOperationsGenerateTest {

    public MatrixOperationsGenerateTest() {
    }

    private static final double EPS = 1.0e-5;
    private List<RealVector> staLocECEFs;

    private static final double[] IN_ARRAY_A = new double[]{
        4.0, 2.0, 3.0};

    private static final double[][] INMATRIXA = new double[][]{
        {12.0, 16.0, 4.0},
        {6.0, 8.0, 2.0},
        {9.0, 12.0, 3.0}};

    private static final double[][] INMATRIXB = new double[][]{
        {2.0, 1.0, 1.0},
        {6.0, 1.0, 1.0},
        {9.0, 1.0, 1.0}};

    private RealMatrix matrixA;
    private RealMatrix matrixB;

    @Before
    public void setUp() {
        this.matrixA = MatrixUtils.createRealMatrix(INMATRIXA);
        this.matrixB = MatrixUtils.createRealMatrix(INMATRIXB);

        this.staLocECEFs = new ArrayList<>();
        //1 - 8014
        staLocECEFs.add(MatrixUtils.createRealVector(new double[]{
            11307223.7, 10476270.2, 21708549.1}));
        //1 - 5190
        staLocECEFs.add(MatrixUtils.createRealVector(new double[]{
            3142790.2, -19372456.4, 17766495.3
        }));
        //2 - 6492
        staLocECEFs.add(MatrixUtils.createRealVector(new double[]{
            -4539465.7, -18103454.0, 18764176.2
        }));
        //1 - 8456
        staLocECEFs.add(MatrixUtils.createRealVector(new double[]{
            -11355067.2, -9093074.6, 22076153.4
        }));
        //2 - 4879
        staLocECEFs.add(MatrixUtils.createRealVector(new double[]{
            5809299.2, -22973553.0, -11630687.0
        }));
        //1 - 8014
        staLocECEFs.add(MatrixUtils.createRealVector(new double[]{
            11306691.0, 10477112.5, 21708421.3
        }));
    }

    @Test
    public void testFlipTB() {

        RealMatrix flipTB = MatrixOperations
                .flipMatrixTB(matrixB);

        double[][] inMatrixX = new double[][]{
            {9.0, 1.0, 1.0},
            {6.0, 1.0, 1.0},
            {2.0, 1.0, 1.0}};

        RealMatrix delta = MatrixUtils.createRealMatrix(
                inMatrixX).subtract(flipTB);

        assertEquals(0.0, delta.getFrobeniusNorm(), EPS);
    }

    @Test
    public void testFlipLR() {

        RealMatrix flipLR = MatrixOperations
                .flipMatrixLR(matrixB);

        double[][] inMatrixX = new double[][]{
            {1.0, 1.0, 2.0},
            {1.0, 1.0, 6.0},
            {1.0, 1.0, 9.0}};

        RealMatrix delta = MatrixUtils.createRealMatrix(
                inMatrixX).subtract(flipLR);

        assertEquals(0.0, delta.getFrobeniusNorm(), EPS);
    }

    @Test
    public void testGenerateMatrixFromList() {
        RealMatrix fromList = MatrixOperations
                .generateMatrixFromList(staLocECEFs);

        double[][] inMatrixX = new double[][]{
            {11307223.7, 10476270.2, 21708549.1},
            {3142790.2, -19372456.4, 17766495.3},
            {-4539465.7, -18103454.0, 18764176.2},
            {-11355067.2, -9093074.6, 22076153.4},
            {5809299.2, -22973553.0, -11630687.0},
            {11306691.0, 10477112.5, 21708421.3},};

        RealMatrix delta = MatrixUtils.createRealMatrix(inMatrixX)
                .subtract(fromList);

        assertEquals(0.0, delta.getFrobeniusNorm(), EPS);
    }

    @Test
    public void testUnpackMatrix() {

        double[] unpackedArray = unpackMatrix(INMATRIXA);

        double[] expectedArray = new double[]{
            12.0, 16.0, 4.0,
            6.0, 8.0, 2.0,
            9.0, 12.0, 3.0};

        double[][] packedMatrix = MatrixOperations.packMatrix(
                expectedArray, 3);

        assertArrayEquals(expectedArray, unpackedArray, EPS);

        double delta = matrixA.subtract(
                MatrixUtils.createRealMatrix(packedMatrix).transpose()).getFrobeniusNorm();

        assertEquals(delta, 0.0, EPS);

    }

    @Test
    public void testMeshGridXY() {

        double[][] meshX = MatrixOperations.meshGridX(IN_ARRAY_A, 3);

        double[][] expectedArrayX = new double[][]{
            {4.0, 4.0, 4.0},
            {2.0, 2.0, 2.0},
            {3.0, 3.0, 3.0}};

        double delta1 = MatrixUtils.createRealMatrix(meshX)
                .subtract(MatrixUtils.createRealMatrix(expectedArrayX))
                .getFrobeniusNorm();

        assertEquals(delta1, 0.0, EPS);

        double[][] meshY = MatrixOperations.meshGridY(IN_ARRAY_A, 3);

        double[][] expectedArrayY = new double[][]{
            {4.0, 2.0, 3.0},
            {4.0, 2.0, 3.0},
            {4.0, 2.0, 3.0}};

        double delta2 = MatrixUtils.createRealMatrix(meshY)
                .subtract(MatrixUtils.createRealMatrix(expectedArrayY))
                .getFrobeniusNorm();

        assertEquals(delta2, 0.0, EPS);

        // ===============================================
        RealVector arrayVector = MatrixUtils.createRealVector(IN_ARRAY_A);
        RealMatrix matrixCol = MatrixOperations
                .replicateMatrixColumns(arrayVector, 3);
        RealMatrix matrixRow = MatrixOperations
                .replicateMatrixRows(arrayVector, 3);

        double delta3 = matrixCol
                .subtract(MatrixUtils.createRealMatrix(expectedArrayX))
                .getFrobeniusNorm();

        assertEquals(delta3, 0.0, EPS);

        double delta4 = matrixRow
                .subtract(MatrixUtils.createRealMatrix(expectedArrayY))
                .getFrobeniusNorm();

        assertEquals(delta4, 0.0, EPS);

    }
}
