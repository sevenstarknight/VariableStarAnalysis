/*
 * Copyright (C) 2018 Kyle Johnston 
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
package fit.astro.vsa.analysis.df.generators;

import fit.astro.vsa.common.utilities.math.linearalgebra.MatrixOperations;
import fit.astro.vsa.common.bindings.math.matrix.UnivariateFunctionMapper;
import fit.astro.vsa.common.utilities.math.linearalgebra.VectorOperations;
import java.util.Arrays;
import java.util.NavigableSet;
import java.util.TreeSet;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.analysis.function.Exp;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston
 */
public class DFOptions {

    public static enum Directions {
        row, column, both
    };

    private final int xDimension;
    private final int yDimension;
    
    private final NavigableSet<Double> xStates;
    private final NavigableSet<Double> yStates;

    private final int[] kernelSize;
    private final double sigmaKernel;
    private final Directions direction;

    private final RealMatrix kernel;

    /**
     *
     * @param numXStates
     * @param numYStates
     * @param kernelSize
     * @param sigmaKernel
     * @param direction
     */
    public DFOptions(int numXStates, int numYStates,
            int[] kernelSize, double sigmaKernel, Directions direction) {

        this.xDimension = numXStates;
        this.yDimension = numYStates;
        
        Double[] doubleArrayX = ArrayUtils.toObject(VectorOperations.linearSpace(1.0, 0.0, numXStates + 1));

        this.xStates = new TreeSet<>(
                Arrays.asList(doubleArrayX));

        Double[] doubleArrayY = ArrayUtils.toObject(VectorOperations.linearSpace(1.0, 0.0, numYStates + 1));

        this.yStates = new TreeSet<>(
                Arrays.asList(doubleArrayY));

        this.kernelSize = kernelSize;
        this.sigmaKernel = sigmaKernel;
        this.direction = direction;

        this.kernel = gaussian2D();
    }

    /**
     *
     * @return
     */
    private RealMatrix gaussian2D() {

        double[] sizTmp = new double[kernelSize.length];
        sizTmp[0] = (kernelSize[0] - 1) / 2;
        sizTmp[1] = (kernelSize[1] - 1) / 2;

        // Generate Grid
        RealVector xVector = MatrixUtils.createRealVector(VectorOperations.linearSpace(sizTmp[0], -sizTmp[0], 1.0));
        RealVector yVector = MatrixUtils.createRealVector(VectorOperations.linearSpace(sizTmp[1], -sizTmp[1], 1.0));

        RealMatrix xMesh = MatrixOperations
                .replicateMatrixRows(xVector, yVector.getDimension());
        RealMatrix yMesh = MatrixOperations
                .replicateMatrixColumns(yVector, xVector.getDimension());

        RealMatrix summa = MatrixOperations.hadamardProduct(xMesh, xMesh).add(
                MatrixOperations.hadamardProduct(yMesh, yMesh));

        // Analytic Function
        RealMatrix expon = summa.scalarMultiply(-1.0 / (2.0 * sigmaKernel * sigmaKernel));

        RealMatrix h = expon.copy();
        h.walkInOptimizedOrder(new UnivariateFunctionMapper(new Exp()));

        // Normalized filter to unit L1 Energy
        double total = MatrixOperations.sumOfElements(h.getData());
        return h.scalarMultiply(1.0 / total);
    }

    public NavigableSet<Double> getxStates() {
        return xStates;
    }

    public NavigableSet<Double> getyStates() {
        return yStates;
    }

    public int[] getKernelSize() {
        return kernelSize;
    }

    public double getSigmaKernel() {
        return sigmaKernel;
    }

    public Directions getDirection() {
        return direction;
    }

    public RealMatrix getKernel() {
        return kernel;
    }

    public int getxDimension() {
        return xDimension;
    }

    public int getyDimension() {
        return yDimension;
    }

    
    
}
