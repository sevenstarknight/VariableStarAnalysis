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
package fit.astro.vsa.utilities.ml.ldaqda;

import fit.astro.vsa.common.bindings.ml.DiscrimantAnalysisMethod;
import static fit.astro.vsa.common.bindings.ml.DiscrimantAnalysisMethod.*;
import fit.astro.vsa.common.bindings.ml.DiscrimantAnalysisResult;
import fit.astro.vsa.common.utilities.math.linearalgebra.MatrixOperations;
import fit.astro.vsa.common.utilities.math.linearalgebra.VectorOperations;
import fit.astro.vsa.common.datahandling.LabelHandling;
import java.util.HashMap;
import java.util.Map;
import org.apache.commons.math3.analysis.function.Power;
import org.apache.commons.math3.analysis.function.Sqrt;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.correlation.Covariance;

/**
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class LDAParamsGenerator implements DiscriminateAnalysisParamsGenerator {

    private final Map<Integer, RealVector> patternArrayTraining;
    private final Map<Integer, String> labelTraining;
    private final DiscrimantAnalysisMethod analysisMethod;

    private final Map<String, Map<Integer, RealVector>> classMembers;
    private final Map<String, Integer> uniqueLabelCount;
    private final int lengthData;
    private final int lengthDims;

    private RealMatrix covMatrix;

    /**
     *
     * @param patternArrayTraining
     * @param labelTraining
     * @param analysisMethod
     */
    public LDAParamsGenerator(
            Map<Integer, RealVector> patternArrayTraining,
            Map<Integer, String> labelTraining,
            DiscrimantAnalysisMethod analysisMethod) {
        this.patternArrayTraining = patternArrayTraining;
        this.labelTraining = labelTraining;
        this.analysisMethod = analysisMethod;

        this.lengthData = patternArrayTraining.size();
        this.lengthDims = patternArrayTraining.values().iterator().next().getDimension();

        this.covMatrix
                = MatrixUtils.createRealMatrix(lengthDims, lengthDims);

        this.classMembers
                = LabelHandling.sortIntoMaps(
                        patternArrayTraining, labelTraining);

        this.uniqueLabelCount
                = LabelHandling.countUniqueClasses(labelTraining);
    }

    /**
     *
     * @return the lda Results (an array of results)
     */
    @Override
    public DiscriminantAnalysis generateParams() {

        //normalize data
        Map<Integer, RealVector> setOfTrainingDataNormal = new HashMap<>();

        
        int dimens = patternArrayTraining.values().iterator().next().getDimension();
        
        // =============
        RealVector xValues = new ArrayRealVector(dimens);
        RealVector x2Values = new ArrayRealVector(dimens);
        for (Integer idx : patternArrayTraining.keySet()) {
            xValues = xValues.add(patternArrayTraining.get(idx));
            x2Values = x2Values.add(patternArrayTraining.get(idx).map(new Power(2.0)));
        }

        RealVector meanX = xValues
                .mapDivide(patternArrayTraining.keySet().size());

        RealVector meanX2 = x2Values.mapDivide(patternArrayTraining.keySet().size());

        RealVector stdX = (meanX2.subtract(meanX.map(new Power(2.0)))).map(new Sqrt());

        for (Integer idx : patternArrayTraining.keySet()) {
            setOfTrainingDataNormal.put(idx, (patternArrayTraining
                    .get(idx).subtract(meanX)).ebeDivide(stdX));
        }
        
        // =================================================================
        switch (analysisMethod) {
            case LDA_GENERAL:
                covMatrix = generalCaseLDA();
                break;
            case LDA_ISOTROPIC:
                covMatrix = isotropicCaseLDA();
                break;
            case LDA_NAIVE:
                covMatrix = naiveCaseLDA();
                break;
            default:
                throw new ArithmeticException("Case not handled");
        }

        //============================================================
        Map<String, DiscrimantAnalysisResult> ldaResults = new HashMap<>();

        for (String currentClass : classMembers.keySet()) {

            // Get the Reduced Set of Vectors
            Map<Integer, RealVector> reducedSet = classMembers.get(currentClass);

            // Generate the Centroids
            RealVector meanCentroid = new ArrayRealVector(
                    reducedSet.values().iterator().next().getDimension());

            for (Integer idx : reducedSet.keySet()) {
                meanCentroid = meanCentroid.add(
                        reducedSet.get(idx).mapDivide((double) reducedSet.size()));
            }

            // Pre Compute the Inv Covariance Matrix
            RealMatrix invCovMatrix
                    = new LUDecomposition(covMatrix).getSolver().getInverse();

            // Pre Compute the Log Det
            double logDeterminant = Math.log(new LUDecomposition(covMatrix).getDeterminant());

            //Store Results
            ldaResults.put(currentClass,
                    new DiscrimantAnalysisResult(reducedSet, meanCentroid,
                            covMatrix, invCovMatrix, logDeterminant,
                            labelTraining.size()));

        }

        return new DiscriminantAnalysis(meanX, stdX, analysisMethod, ldaResults, uniqueLabelCount);
    }

    /**
     *
     * @param patternMatrix
     * @param labels
     * @param uniques
     * <p>
     * @return Real Matrix, covariance matrix
     */
    private RealMatrix generalCaseLDA() {

        for (String currentClass : classMembers.keySet()) {

            Map<Integer, RealVector> reducedSet = classMembers.get(currentClass);

            RealMatrix reducedSetMatrix
                    = MatrixOperations.generateMatrixFromMap(reducedSet);

            // Mean Covariance Matrix (Full) Over All Points
            Covariance covariance = new Covariance(reducedSetMatrix);
            covMatrix = covMatrix.add(
                    covariance.getCovarianceMatrix().scalarMultiply(
                            (double) reducedSet.size() / (double) lengthData));

        }

        return covMatrix;
    }

    /**
     *
     * @param patternMatrix
     * @param labels
     * @param uniques
     * <p>
     * @return Real Matrix, covariance matrix
     */
    private RealMatrix naiveCaseLDA() {

        for (String currentClass : classMembers.keySet()) {

            Map<Integer, RealVector> reducedSet = classMembers.get(currentClass);

            RealMatrix reducedSetMatrix
                    = MatrixOperations.generateMatrixFromMap(reducedSet);

            RealMatrix naiveCovMatrix
                    = MatrixUtils.createRealMatrix(lengthDims, lengthDims);

            // Variance along each dimension, diagonal covariance matrix
            for (int j = 0; j < lengthDims; j++) {
                double std
                        = VectorOperations.std(reducedSetMatrix.getColumnVector(j));
                naiveCovMatrix.setEntry(j, j, std * std);
            }

            covMatrix = covMatrix.add(
                    naiveCovMatrix.scalarMultiply(
                            (double) reducedSet.size() / (double) lengthData));

        }

        return covMatrix;
    }

    /**
     *
     * @param patternMatrix
     * @param labels
     * @param uniques
     * <p>
     * @return Real Matrix, covariance matrix
     */
    private RealMatrix isotropicCaseLDA() {

        for (String currentClass : classMembers.keySet()) {

            Map<Integer, RealVector> reducedSet = classMembers.get(currentClass);

            RealMatrix reducedSetMatrix
                    = MatrixOperations.generateMatrixFromMap(reducedSet);

            RealMatrix naiveCovMatrix
                    = MatrixUtils.createRealMatrix(lengthDims, lengthDims);

            for (int j = 0; j < lengthDims; j++) {
                double std
                        = VectorOperations.std(reducedSetMatrix.getColumnVector(j));
                naiveCovMatrix.setEntry(j, j, std * std);
            }

            // Mean Variance over dimension, over all points, diagonal 
            // covariance
            covMatrix = covMatrix.add(
                    naiveCovMatrix.scalarMultiply(
                            (double) reducedSet.size() / (double) lengthData));

        }

        double varPool = 0;
        for (int j = 0; j < lengthDims; j++) {
            varPool = covMatrix.getEntry(j, j) + varPool;
        }

        covMatrix = MatrixUtils.createRealIdentityMatrix(lengthDims).
                scalarMultiply((double) varPool / (double) lengthDims);

        return covMatrix;
    }

    /**
     * @return the analysisMethod
     */
    public DiscrimantAnalysisMethod getAnalysisMethod() {
        return analysisMethod;
    }

}
