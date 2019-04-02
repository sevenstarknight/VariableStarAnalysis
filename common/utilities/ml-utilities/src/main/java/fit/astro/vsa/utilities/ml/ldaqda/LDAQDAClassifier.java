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
import fit.astro.vsa.common.utilities.math.support.SortingOperations;
import fit.astro.vsa.common.bindings.ml.ClassificationResult;
import java.util.HashMap;
import java.util.Map;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * Friedman, J., Hastie, T., & Tibshirani, R. (2001). The elements of
 * statistical learning (Vol. 1, No. 10). New York, NY, USA:: Springer series in
 * statistics.
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class LDAQDAClassifier {

    private final Map<Integer, RealVector> setOfTrainingData;
    private final Map<Integer, String> setOfTrainingClasses;

    /**
     *
     * @param setOfTrainingData
     * @param setOfTrainingClasses
     */
    public LDAQDAClassifier(Map<Integer, RealVector> setOfTrainingData,
            Map<Integer, String> setOfTrainingClasses) {
        this.setOfTrainingData = setOfTrainingData;
        this.setOfTrainingClasses = setOfTrainingClasses;
    }

    /**
     *
     * @param analysisMethod
     * @return
     */
    public DiscriminantAnalysis generateDiscriminateAnalysis(
            DiscrimantAnalysisMethod analysisMethod) {

        switch (analysisMethod) {
            case LDA_GENERAL:
            case LDA_ISOTROPIC:
            case LDA_NAIVE:

                LDAParamsGenerator lda = new LDAParamsGenerator(
                        setOfTrainingData,
                        setOfTrainingClasses, analysisMethod);

                return lda.generateParams();

            case QDA_GENERAL:
            case QDA_ISOTROPIC:
            case QDA_NAIVE:
                QDAParamsGenerator qda = new QDAParamsGenerator(
                        setOfTrainingData,
                        setOfTrainingClasses, analysisMethod);
                return qda.generateParams();

            default:
                throw new ArithmeticException("Execute");
        }
    }

    /**
     *
     * @param ldaqda
     * @param inputPatternMap
     * <p>
     * @return
     */
    public ClassificationResult execute(
            DiscriminantAnalysis ldaqda,
            Map<Integer, RealVector> inputPatternMap) {

        // run
        Map<Integer, String> labelEstimate = new HashMap<>();
        Map<Integer, Map<String, Double>> labelPostProb = new HashMap<>();

        for (Integer idx : inputPatternMap.keySet()) {

            Map<String, Double> classDiscrimMap
                    = SoftClassifier(inputPatternMap.get(idx), ldaqda);

            // Find Max 
            Map<String, Double> sortedProbs
                    = SortingOperations.sortByDecendingValue(classDiscrimMap);
            String selectedClass = sortedProbs.keySet().iterator().next();

            labelEstimate.put(idx, selectedClass);
            labelPostProb.put(idx, sortedProbs);
        }

        return new ClassificationResult(labelEstimate, labelPostProb,
                ldaqda.getUniqueLabelCount());
    }

    /**
     *
     * @param inputPatternMap
     * @param ldaqda
     * <p>
     * @return
     */
    private static Map<Integer, String> HardClassifier(
            Map<Integer, RealVector> inputPatternMap,
            DiscriminantAnalysis ldaqda) {

        Map<Integer, String> classEstimate = new HashMap<>();

        for (Integer idx : inputPatternMap.keySet()) {

            Map<String, Double> classDiscrimMap
                    = SoftClassifier(inputPatternMap.get(idx), ldaqda);

            // Find Max 
            Map<String, Double> sortedProbs
                    = SortingOperations.sortByDecendingValue(classDiscrimMap);

            String selectedClass = sortedProbs.keySet().iterator().next();

            classEstimate.put(idx, selectedClass);
        }
        return classEstimate;
    }

    /**
     *
     * @param currentPatternTmp
     * @param ldaqda
     * <p>
     * @return
     */
    private static Map<String, Double> SoftClassifier(
            RealVector currentPatternTmp,
            DiscriminantAnalysis ldaqda) {

        RealVector meanX = ldaqda.getMeanX();
        RealVector stdX = ldaqda.getStdX();

        RealVector currentPattern = (currentPatternTmp
                .subtract(meanX)).ebeDivide(stdX);

        Map<String, Double> classDiscrimMap = new HashMap<>();

        double maxDiscrim = Double.MIN_VALUE;

        for (String currentLabel : ldaqda.getParams().keySet()) {

            DiscrimantAnalysisResult currentResult
                    = ldaqda.getParams().get(currentLabel);

            RealVector mu = currentResult.getMeanCentroid();
            RealMatrix invCov = currentResult.getInvCovMatrix();
            double constant = currentResult.getConstant();

            RealVector delta = currentPattern.subtract(mu);

            // which is also the postieror probability
            double discriminateFunction = constant
                    - 0.5 * delta.dotProduct(invCov.operate(delta));

            if (maxDiscrim < discriminateFunction) {
                maxDiscrim = discriminateFunction;
            }

            classDiscrimMap.put(currentLabel, discriminateFunction);
        }

        for (String currentLabel : ldaqda.getParams().keySet()) {
            classDiscrimMap.put(currentLabel, Math.exp(
                    classDiscrimMap.get(currentLabel) - maxDiscrim));
        }

        return classDiscrimMap;
    }

    /**
     *
     * @param ldaqda
     * @param inputPatternMap
     * @param inputClassMap
     * <p>
     * @return
     */
    public static double estimateMisclassificationRate(
            DiscriminantAnalysis ldaqda,
            Map<Integer, RealVector> inputPatternMap,
            Map<Integer, String> inputClassMap) {

        double misclassifications = 0;

        for (Integer idx : inputPatternMap.keySet()) {
            RealVector currentPattern = inputPatternMap.get(idx);
            Map<String, Double> classDiscrimMap
                    = SoftClassifier(currentPattern, ldaqda);

            // Find Max 
            Map<String, Double> sortedProbs
                    = SortingOperations.sortByDecendingValue(classDiscrimMap);
            String selectedClass = sortedProbs.keySet().iterator().next();

            if (!inputClassMap.get(idx).equalsIgnoreCase(selectedClass)) {
                // Terminal Node Found 
                misclassifications++;
            }
        }

        return misclassifications / (double) inputPatternMap.size();
    }

    /**
     * To run the classifier, the whole training dataset is not needed, only the
     * LDAQDA objected generated by the training operation.
     * <p>
     * @param ldaqda
     * @param pattern
     * <p>
     * @return
     */
    public static String estimateClassLabel(
            DiscriminantAnalysis ldaqda,
            RealVector pattern) {
        Map<String, Double> classDiscrimMap
                = SoftClassifier(pattern, ldaqda);

        // Find Max 
        Map<String, Double> sortedProbs
                = SortingOperations.sortByDecendingValue(classDiscrimMap);

        return sortedProbs.keySet().iterator().next();
    }

}
