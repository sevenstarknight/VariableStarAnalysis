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
package fit.astro.vsa.utilities.ml.ecva;

import fit.astro.vsa.common.utilities.math.linearalgebra.MatrixOperations;
import fit.astro.vsa.common.utilities.math.linearalgebra.VectorOperations;
import fit.astro.vsa.common.datahandling.LabelHandling;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.commons.math3.analysis.function.Sqrt;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.apache.commons.math3.stat.correlation.Covariance;
import org.nd4j.linalg.api.blas.impl.BaseLapack;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.cpu.nativecpu.blas.CpuLapack;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class SubECVA {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(SubECVA.class);

    private final Map<String, Map<Integer, RealVector>> classMembersTraining;

    private RealMatrix sWithin;
    private final RealMatrix sBetween;
    private final RealMatrix sTotal;

    private final Map<Integer, RealMatrix> bWeights;

    private final Map<Integer, RealMatrix> cvaWeights;
    private final Map<Integer, Map<Integer, RealVector>> projScores;
    private final Map<Integer, Double> vMap;

    private final BaseLapack lapack = new CpuLapack();

    public SubECVA(Map<Integer, RealVector> classPattern,
            Map<Integer, String> classLabel,
            int minNumberOfComponents, int maxNumberOfComponents) {

        int dimension = classPattern.values().iterator().next().getDimension();

        this.classMembersTraining = LabelHandling.sortIntoMaps(
                classPattern, classLabel);

        // ============================================
        //Find S within Classes
        Map<String, RealVector> meanOfClasses = new HashMap<>(
                classMembersTraining.keySet().size());

        this.sWithin = new BlockRealMatrix(dimension, dimension);
        for (String e : classMembersTraining.keySet()) {
            Map<Integer, RealVector> tmpList = classMembersTraining.get(e);

            RealMatrix xSub = MatrixOperations.generateMatrixFromMap(tmpList);

            RealVector meanOColumn = MatrixOperations.dimensionalMean(xSub, Boolean.TRUE);
            meanOfClasses.put(e, meanOColumn);

            Covariance covarianceSub = new Covariance(xSub);
            RealMatrix withinCovar = covarianceSub.getCovarianceMatrix();

            sWithin = sWithin.add(withinCovar
                    .scalarMultiply((double) classMembersTraining.get(e).size() - 1.0));
        }

        // ============================================
        // Find S For All
        RealMatrix xSub = MatrixOperations.generateMatrixFromMap(classPattern);
        Covariance covarianceSub = new Covariance(xSub);
        RealMatrix totalCovar = covarianceSub.getCovarianceMatrix();

        this.sTotal = totalCovar.scalarMultiply((double) classPattern.size() - 1.0);
        // ============================================
        // Find S between
        this.sBetween = sTotal.subtract(sWithin);

        // ============================================
        RealMatrix xAll = MatrixOperations.generateMatrixFromMap(classPattern);
        RealVector mAll = MatrixOperations.dimensionalMean(xAll, Boolean.TRUE);

        RealMatrix y = MatrixUtils.createRealMatrix(dimension,
                classMembersTraining.keySet().size());

        int counter = 0;
        for (String e : classMembersTraining.keySet()) {
            y.setColumnVector(counter, meanOfClasses.get(e).subtract(mAll));
            counter++;
        }

        // Special case of two groups
        if (classMembersTraining.keySet().size() == 2) {
            List<String> tmpString = new ArrayList<>(
                    classMembersTraining.keySet());

            RealVector m1 = MatrixOperations.dimensionalMean(
                    MatrixOperations.generateMatrixFromMap(classMembersTraining
                            .get(tmpString.get(0))), Boolean.TRUE);

            RealVector m2 = MatrixOperations.dimensionalMean(
                    MatrixOperations.generateMatrixFromMap(classMembersTraining
                            .get(tmpString.get(1))), Boolean.TRUE);

            y.setColumnVector(0, m1.subtract(m2));
        }

        // ===============================================================
        /**
         * Calculate the weights through PLS models. Note: with e.g. six groups,
         * six y-variables (and not five) are used to obtain an equal weighting
         * of directions in PLS2. It seems that NO mean centering gives slightly
         * better classification
         */
        this.bWeights = estimateWeights(sWithin, y, maxNumberOfComponents);

        Map<Integer, RealMatrix> preCVAWeights = new HashMap<>(bWeights);

        /**
         * At this point the number of CVAs will be equal to number of groups
         * except if the input number of groups is 2, then the number of CVAs is
         * 1 and no elimination is necessary
         */
        // ===============================================================
        // Order the weights according to optimization criterion and eliminate the poorest
        if (classMembersTraining.keySet().size() > 2
                && classMembersTraining.keySet().size()
                <= classPattern.size()) {

            for (int jdx = minNumberOfComponents - 1; jdx < maxNumberOfComponents; jdx++) {

                List<Double> optCrit = new ArrayList<>(classMembersTraining.size());

                for (int kdx = 0; kdx < classMembersTraining.size(); kdx++) {

                    RealVector wTemp = preCVAWeights.get(jdx).getColumnVector(kdx);

                    double top = wTemp.dotProduct(sBetween.operate(wTemp));
                    double bottom = wTemp.dotProduct(sWithin.operate(wTemp));

                    optCrit.add(top / bottom);

                }

                List<Double> nStore = new ArrayList<>(optCrit);

                // Decending
                Collections.sort(nStore);
                Collections.reverse(nStore);

                // Flip sign so that the average element is positive;
                RealMatrix tmpCVA = MatrixUtils.createRealMatrix(
                        preCVAWeights.get(jdx).getData());

                for (int n = 0; n < nStore.size(); n++) {

                    RealVector tmpV = preCVAWeights.get(jdx).getColumnVector(n);

                    if (VectorOperations.summationOfElements(tmpV) < 0) {
                        tmpV = tmpV.mapMultiply(-1.0);
                    }

                    tmpCVA.setColumnVector(nStore.indexOf(optCrit.get(n)), tmpV);

                }

                // Leave out the poorest performing
                int[] selectedColumns = VectorOperations.linearSpace(
                        tmpCVA.getColumnDimension() - 2, 0, 1);

                int[] selectedRows = VectorOperations.linearSpace(
                        tmpCVA.getRowDimension() - 1, 0, 1);

                preCVAWeights.put(jdx, tmpCVA.getSubMatrix(selectedRows, selectedColumns));

            }

        } else {

            for (int jdx = minNumberOfComponents - 1; jdx < maxNumberOfComponents; jdx++) {

                RealMatrix tmpW = preCVAWeights.get(jdx);
                int[] selectedColumns = VectorOperations.linearSpace(
                        jdx - 1, 0, 1);

                int[] selectedRows = VectorOperations.linearSpace(
                        tmpW.getRowDimension() - 1, 0, 1);

                preCVAWeights.put(jdx, tmpW.getSubMatrix(selectedRows, selectedColumns));
            }
        }

        // ===============================================================
        // Collect, rotate and normalize weights ("loadings") for each component
        this.cvaWeights = new HashMap<>(maxNumberOfComponents - minNumberOfComponents + 1);

        for (int jdx = minNumberOfComponents - 1; jdx < maxNumberOfComponents; jdx++) {

            // Rotate weights to wi*Swithin*wj=0
            RealMatrix tmpCVA = preCVAWeights.get(jdx);

            RealMatrix input = tmpCVA.transpose().multiply(sWithin.multiply(tmpCVA));

            INDArray nd = new NDArray(input.getData());

            int nRows = nd.rows();
            int nColumns = nd.columns();

            INDArray s = Nd4j.zeros(1, nRows);
            INDArray u = Nd4j.zeros(nRows, nRows);
            INDArray vt = Nd4j.zeros(nColumns, nColumns);

            lapack.gesvd(nd, s, u, vt);

            RealMatrix v = new Array2DRowRealMatrix(vt.transpose().toDoubleMatrix());

            tmpCVA = tmpCVA.multiply(v);

            RealMatrix tmpMatrix = tmpCVA.transpose().multiply(sWithin.multiply(tmpCVA));
            RealVector onesVector = MatrixUtils.createRealVector(VectorOperations.ones(tmpCVA.getColumnDimension()));

            RealVector k = onesVector.ebeDivide(MatrixOperations.getDiag(tmpMatrix));

            onesVector = MatrixUtils.createRealVector(VectorOperations.ones(tmpCVA.getRowDimension()));

            RealMatrix newMatrix = onesVector.outerProduct(k.map(new Sqrt()));

            cvaWeights.put(jdx, MatrixOperations.hadamardProduct(tmpCVA, newMatrix));
        }

        //==================================================================
        /**
         * Calculate canonical variates - the "scores" Mean centering of X.raw
         * is performed as in the standard CVA (manoval)
         */
        RealMatrix subMC = new Array2DRowRealMatrix(classLabel.size(), dimension);
        counter = 0;
        for (Integer idx : classLabel.keySet()) {
            subMC.setRowVector(counter, classPattern.get(idx).subtract(mAll));
            counter++;
        }

        this.projScores = new HashMap<>(maxNumberOfComponents - minNumberOfComponents + 1);

        for (int jdx = minNumberOfComponents - 1; jdx < maxNumberOfComponents; jdx++) {

            Map<Integer, RealVector> projectMap = new HashMap<>();
            for (Integer idx : classPattern.keySet()) {
                RealVector meanSub = classPattern.get(idx).subtract(mAll);
                projectMap.put(idx, cvaWeights.get(jdx).transpose().operate(meanSub));
            }

            projScores.put(jdx, projectMap);
        }

        //==================================================================
        /**
         * Calculate optimization criterion
         */
        this.vMap = new HashMap<>(maxNumberOfComponents - minNumberOfComponents + 1);

        for (int jdx = minNumberOfComponents - 1; jdx < maxNumberOfComponents; jdx++) {

            RealMatrix wTemp = cvaWeights.get(jdx);

            RealMatrix top = (wTemp.transpose().multiply(sBetween.multiply(wTemp)));
            RealMatrix bottom = wTemp.transpose().multiply(sWithin.multiply(wTemp));

            LUDecomposition ludt = new LUDecomposition(top);
            LUDecomposition ludb = new LUDecomposition(bottom);

            vMap.put(jdx, ludt.getDeterminant() / ludb.getDeterminant());

        }

    }

    private Map<Integer, RealMatrix> estimateWeights(
            RealMatrix sWithin, RealMatrix y, int levels) {

        RealMatrix sMatrix = sWithin.transpose().multiply(y);

        RealMatrix vMatrix = MatrixUtils.createRealMatrix(
                sMatrix.getRowDimension(), 1);
        RealMatrix qMatrix = MatrixUtils.createRealMatrix(
                sMatrix.getColumnDimension(), levels);
        RealMatrix rMatrix = MatrixUtils.createRealMatrix(
                sWithin.getRowDimension(), levels);

        for (int i = 0; i < levels; i++) {

            // ==========================
            SingularValueDecomposition xsvd
                    = new SingularValueDecomposition(sMatrix);

            RealVector qVector = xsvd.getV().getColumnVector(0);

            RealVector rVector = sMatrix.operate(qVector);
            RealVector tVector = sWithin.operate(rVector);

            double normOfT = tVector.getNorm();

            RealVector normT = tVector.mapDivide(normOfT);
            RealVector normRVector = rVector.mapDivide(normOfT);

            RealVector pVector = sWithin.transpose().operate(normT);
            qVector = y.transpose().operate(normT);

            RealMatrix pMatrix = MatrixUtils.createRealMatrix(
                    pVector.getDimension(), 1);
            pMatrix.setColumnVector(0, pVector);

            RealVector vVector = new ArrayRealVector(pVector);

            if (i > 0) {
                RealMatrix tmpMatrix = vMatrix.multiply(
                        vMatrix.transpose().multiply(pMatrix));
                vVector = vVector.subtract(tmpMatrix.getColumnVector(0));
            }

            vVector = vVector.mapDivide(vVector.getNorm());
            RealMatrix vMatrixTmp = MatrixUtils.createRealMatrix(
                    vVector.getDimension(), 1);
            vMatrixTmp.setColumnVector(0, vVector);

            sMatrix = sMatrix.subtract(vMatrixTmp.multiply(
                    vMatrixTmp.transpose().multiply(sMatrix)));

            // update Arrays
            if (i == 0) {
                vMatrix.setColumnVector(i, vVector);
            } else {
                RealMatrix tmp = MatrixUtils.createRealMatrix(vMatrix.getData());
                vMatrix = MatrixUtils.createRealMatrix(tmp.getRowDimension(), i + 1);
                vMatrix.setSubMatrix(tmp.getData(), 0, 0);
                vMatrix.setColumnVector(i, vVector);
            }
            qMatrix.setColumnVector(i, qVector);
            rMatrix.setColumnVector(i, normRVector);

        }

        // ===============================================
        int[] selectedRowsR = VectorOperations.linearSpace(
                rMatrix.getRowDimension() - 1, 0, 1);
        int[] selectedRowsQ = VectorOperations.linearSpace(
                qMatrix.getRowDimension() - 1, 0, 1);

        Map<Integer, RealMatrix> bOut = new HashMap<>();

        for (int i = 0; i < levels; i++) {

            int[] selectedColumns = VectorOperations.linearSpace(
                    i, 0, 1);

            RealMatrix subR = rMatrix.getSubMatrix(
                    selectedRowsR, selectedColumns);
            RealMatrix subQT = qMatrix.getSubMatrix(
                    selectedRowsQ, selectedColumns).transpose();

            bOut.put(i, subR.multiply(subQT));
        }

        return bOut;
    }

    /**
     * @return the sWithin
     */
    public RealMatrix getsWithin() {
        return sWithin;
    }

    /**
     * @return the sBetween
     */
    public RealMatrix getsBetween() {
        return sBetween;
    }

    /**
     * @return the sTotal
     */
    public RealMatrix getsTotal() {
        return sTotal;
    }

    /**
     * @return the bWeights
     */
    public Map<Integer, RealMatrix> getbWeights() {
        return bWeights;
    }

    /**
     * @return the cvaWeights
     */
    public Map<Integer, RealMatrix> getCvaWeights() {
        return cvaWeights;
    }

    /**
     * @return the projScores
     */
    public Map<Integer, Map<Integer, RealVector>> getProjScores() {
        return projScores;
    }

    /**
     * @return the vMap
     */
    public Map<Integer, Double> getvMap() {
        return vMap;
    }

    /**
     * @return the classMembersTraining
     */
    public Map<String, Map<Integer, RealVector>> getClassMembersTraining() {
        return classMembersTraining;
    }

}
