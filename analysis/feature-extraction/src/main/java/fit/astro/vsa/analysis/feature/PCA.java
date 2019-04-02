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
package fit.astro.vsa.analysis.feature;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import fit.astro.vsa.common.utilities.math.linearalgebra.VectorOperations;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.apache.commons.math3.stat.correlation.Covariance;

/**
 * PCA - Principal Component Analysis Pearson, K. (1901). "On Lines and Planes
 * of Closest Fit to Systems of Points in Space" (PDF). Philosophical Magazine.
 * 2 (11): 559–572. doi:10.1080/14786440109462720.
 * <p>
 * Jolliffe I.T. Principal Component Analysis, Series: Springer Series in
 * Statistics, 2nd ed., Springer, NY, 2002, XXIX, 487 p. 28 illus. ISBN
 * 978-0-387-95442-4
 *
 * <p>
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class PCA {

    // ====================== Inputs
    private final int dimData;                      //original dimensionality of the input vectors
    private final Map<Integer, RealVector> data;    // input data-vectors

    // ====================== Interior
    private RealVector mean;                // translation vector
    private List<RealVector> centeredData;  // centered data
    private RealMatrix cvmat;               // (variance)-covariance-matrix

    private RealMatrix dataMatrix;

    // ====================== Outputs
    private SingularValueDecomposition decomposition;   // eigenvalue decomposition 
    private RealMatrix vMatrix;                         // eigenvector matrix 
    private double[] eValues;                           // eigenvalue array

    private RealMatrix vMatrixSorted;                   // eigenvector matrix 
    private double[] eValuesSorted;                     // eigenvalue array

    /**
     *
     * @param data
     */
    public PCA(Map<Integer, RealVector> data) {
        this.data = data;
        this.dimData = data.get(0).getDimension();
        centerData();
        computeCovarianceMatrix();
        updatePCAMatrix();
    }

    /**
     * center the input data prior transformed
     */
    private void centerData() {

        // compute mean
        mean = new ArrayRealVector(dimData);
        for (RealVector entry : data.values()) {
            mean = mean.add(entry);
        }

        mean = mean.mapDivide(data.size());

        // center data (subtract mean) -> mean is at origin now
        centeredData = new ArrayList<>();
        for (RealVector entry : data.values()) {
            centeredData.add(entry.subtract(mean));
        }
    }

    /**
     * estimate the covariance matrix based on the input data
     */
    private void computeCovarianceMatrix() {
        // NxN-matrix
        //  . symetric, positive definite or positive semi-definite
        //    every square symmetric matrix is orthogonally (orthonormally) diagonalisable.
        //     --> S = E D E-transpose
        //  . diagonal -> variances
        //  . off-diagonal -> co-variances (... how well correlated two variables are)
        //  1. Maximise the signal, measured by variance (maximise the diagonal entries)
        //  2. Minimise the covariance between variables (minimise the off-diagonal entries)

        dataMatrix = MatrixUtils.createRealMatrix(centeredData.size(), dimData);

        int counter = 0;
        for (RealVector entry : centeredData) {
            dataMatrix.setRowVector(counter, entry);
            counter++;
        }
        Covariance covarianceEst = new Covariance(dataMatrix);
        cvmat = covarianceEst.getCovarianceMatrix();
    }

    /**
     *
     */
    private void updatePCAMatrix() {
        decomposition = new SingularValueDecomposition(dataMatrix);

        vMatrix = decomposition.getV();
        eValues = decomposition.getSingularValues();

        // create objects for sorting
        // columns are eigenvectors ... principal components
        Integer[] idx = ArrayUtils.toObject(VectorOperations.linearSpace(eValues.length - 1, 0, 1));

        Arrays.sort(idx, (Integer i1, Integer i2)
                -> Double.compare(eValues[i1], eValues[i2]));

        Collections.reverse(Arrays.asList(idx));

        // eValues[idx[i]] is the sorted eiginevalue at index i
        // colors[idx[i]] is the sorted color at index i
        eValuesSorted = new double[idx.length];
        vMatrixSorted = MatrixUtils.createRealMatrix(vMatrix.getData());
        for (int i = 0; i < idx.length; i++) {
            eValuesSorted[i] = eValues[idx[i]];
            vMatrixSorted.setColumn(i, vMatrix.getColumn(idx[i]));
        }

    }

    /**
     * Pull the data from the transformed input data (transformed by the PCA)
     *
     * @param intDimensions
     * @return
     */
    public List<RealVector> getTransformedData(int intDimensions) {

        List<RealVector> transformedData = new ArrayList<>();

        for (RealVector centered : centeredData) {

            RealMatrix centeredMatrix = MatrixUtils.createRowRealMatrix(centered.toArray());
            RealMatrix vReducedMatrix = vMatrixSorted.getSubMatrix(
                    0, vMatrixSorted.getRowDimension() - 1,
                    0, intDimensions - 1);

            RealMatrix klt = vReducedMatrix.transpose().multiply(centeredMatrix.transpose());
            transformedData.add(klt.getColumnVector(0));
        }

        // =============================================
        RealMatrix matrixSet = MatrixUtils.createRealMatrix(
                centeredData.size(), intDimensions);
        int counter = 0;
        for (RealVector entry : transformedData) {
            matrixSet.setRowVector(counter, entry);
            counter++;
        }

        Covariance covarianceEst = new Covariance(matrixSet);
        cvmat = covarianceEst.getCovarianceMatrix();

        return transformedData;
    }

    /**
     * Return covariance matrix based on PCA
     *
     * @return
     */
    public RealMatrix getCvmat() {
        return cvmat;
    }

}
