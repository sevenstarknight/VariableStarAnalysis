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
public class QDAParamsGenerator implements DiscriminateAnalysisParamsGenerator {

    private final Map<Integer, RealVector> patternArrayTraining;
    private final Map<Integer, String> labelTraining;
    private final DiscrimantAnalysisMethod analysisMethod;
    private final Map<String, Integer> uniqueLabelCount;

    /**
     *
     * @param patternArrayTraining
     * @param labelTraining
     * @param analysisMethod
     */
    public QDAParamsGenerator(
            Map<Integer, RealVector> patternArrayTraining,
            Map<Integer, String> labelTraining,
            DiscrimantAnalysisMethod analysisMethod) {
        this.patternArrayTraining = patternArrayTraining;
        this.labelTraining = labelTraining;
        this.analysisMethod = analysisMethod;

        this.uniqueLabelCount
                = LabelHandling.countUniqueClasses(labelTraining);
    }

    /**
     *
     * <p>
     * @return
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
        
        
        Map<String, DiscrimantAnalysisResult> qdaResults = new HashMap<>();

        RealMatrix covMatrix;

        Map<String, Map<Integer, RealVector>> classMembers
                = LabelHandling.sortIntoMaps(
                        patternArrayTraining, labelTraining);

        for (String currentClass : classMembers.keySet()) {

            Map<Integer, RealVector> reducedSet = classMembers.get(currentClass);

            switch (analysisMethod) {
                case QDA_GENERAL:
                    covMatrix = generalCaseQDA(reducedSet);
                    break;
                case QDA_ISOTROPIC:
                    covMatrix = isotropicCaseQDA(reducedSet);
                    break;
                case QDA_NAIVE:
                    covMatrix = naiveCaseQDA(reducedSet);
                    break;
                default:
                    throw new ArithmeticException("Case not handled");
            }

            // Prep Result
            RealVector meanCentroid = new ArrayRealVector(reducedSet.values()
                    .iterator().next().getDimension());

            for (RealVector currentSet : reducedSet.values()) {
                meanCentroid = meanCentroid.add(
                        currentSet.mapDivide((double) reducedSet.size()));
            }

            RealMatrix invCovMatrix
                    = new LUDecomposition(covMatrix).getSolver().getInverse();

            double logDeterminant = Math.log(
                    new LUDecomposition(covMatrix).getDeterminant());

            // Store Results
            qdaResults.put(currentClass,
                    new DiscrimantAnalysisResult(
                            reducedSet, meanCentroid,
                            covMatrix, invCovMatrix, logDeterminant,
                            (double) labelTraining.size()));

        }

        return new DiscriminantAnalysis(meanX, stdX,
                analysisMethod, qdaResults, uniqueLabelCount);
    }

    /**
     *
     * @param patternMatrix
     * @param labels
     * @param uniques
     * <p>
     * @return Real Matrix, covariance matrix
     */
    private RealMatrix generalCaseQDA(
            Map<Integer, RealVector> reducedSet) {
        RealMatrix reducedSetMatrix
                = MatrixOperations.generateMatrixFromMap(reducedSet);

        Covariance covariance = new Covariance(reducedSetMatrix);
        return covariance.getCovarianceMatrix();

    }

    /**
     *
     * @param patternMatrix
     * @param labels
     * @param uniques
     * <p>
     * @return Real Matrix, covariance matrix
     */
    private RealMatrix naiveCaseQDA(
            Map<Integer, RealVector> reducedSet) {

        int lengthDims = reducedSet.values().iterator().next().getDimension();
        RealMatrix reducedSetMatrix
                = MatrixOperations.generateMatrixFromMap(reducedSet);

        RealMatrix naiveCovMatrix
                = MatrixUtils.createRealMatrix(lengthDims, lengthDims);

        for (int j = 0; j < lengthDims; j++) {
            double std = VectorOperations.std(reducedSetMatrix.getColumnVector(j));
            naiveCovMatrix.setEntry(j, j, std * std);
        }

        return naiveCovMatrix;

    }

    /**
     *
     * @param patternMatrix
     * @param labels
     * @param uniques
     * <p>
     * @return Real Matrix, covariance matrix
     */
    private RealMatrix isotropicCaseQDA(
            Map<Integer, RealVector> reducedSet) {
        int lengthDims = reducedSet.values().iterator().next().getDimension();

        RealMatrix reducedSetMatrix
                = MatrixOperations.generateMatrixFromMap(reducedSet);

        RealVector isoVector = new ArrayRealVector(lengthDims);

        for (int j = 0; j < lengthDims; j++) {
            double std = VectorOperations.std(reducedSetMatrix.getColumnVector(j));
            isoVector.setEntry(j, std * std);
        }

        double estVar = VectorOperations.mean(isoVector);

        return MatrixUtils.createRealIdentityMatrix(lengthDims).
                scalarMultiply(estVar);

    }

    /**
     * @return the analysisMethod
     */
    public DiscrimantAnalysisMethod getAnalysisMethod() {
        return analysisMethod;
    }

}
