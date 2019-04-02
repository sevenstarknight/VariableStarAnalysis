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
package fit.astro.vsa.utilities.ml.lrc;

import fit.astro.vsa.utilities.ml.utils.SupportingFunctionality;
import fit.astro.vsa.common.utilities.math.handling.exceptions.IterationTimeOutException;
import fit.astro.vsa.common.utilities.math.linearalgebra.MatrixOperations;
import fit.astro.vsa.common.utilities.math.linearalgebra.VectorOperations;
import fit.astro.vsa.common.bindings.ml.ClassificationResult;
import fit.astro.vsa.utilities.ml.performance.ClassifierPerformance;
import fit.astro.vsa.common.datahandling.training.TrainCrossTestGenerator;
import fit.astro.vsa.common.datahandling.LabelHandling;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Friedman, J., Hastie, T., & Tibshirani, R. (2001). The elements of
 * statistical learning (Vol. 1, No. 10). New York, NY, USA:: Springer series in
 * statistics.
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class LRClassifierGenerator {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(LRClassifierGenerator.class);
    // ===================================================================
    // Input Values
    private final Map<Integer, RealVector> setOfPatterns;
    private final Map<Integer, String> setOfClasses;

    private final TrainCrossTestGenerator trainTest;

    // ===================================================================
    // Internal Values
    private final Map<String, Integer> uniqueLabelCount;

    private final int dimension;
    private int number;

    private Map<String, Map<Integer, RealVector>> classMembers;

    private Map<Integer, RealVector> newSetOfTrainingPatterns;
    private Map<Integer, String> newSetOfTrainingClasses;
    private Map<Integer, RealVector> newSetOfCrossvalPatterns;
    private Map<Integer, String> setOfCrossvalClasses;

    private Map<Integer, RealVector> yResponse;

    private RealMatrix xTilda;
    private Map<String, RealVector> betaParams;

    // ===================================================================
    // Adjustable Parameters
    private double regularizationValue = 0.02;

    /**
     *
     * @param setOfPatterns
     * @param setOfClasses
     * @param trainTest
     */
    public LRClassifierGenerator(
            Map<Integer, RealVector> setOfPatterns,
            Map<Integer, String> setOfClasses,
            TrainCrossTestGenerator trainTest) {

        this.setOfPatterns = setOfPatterns;
        this.setOfClasses = setOfClasses;

        this.trainTest = trainTest;

        // == Counts 
        this.uniqueLabelCount = LabelHandling.countUniqueClasses(setOfClasses);
        this.dimension = setOfPatterns.values().iterator().next().getDimension();

    }

    // ====================================================================
    /**
     *
     * @return @throws IterationTimeOutException
     */
    public LRC generateLRC() throws IterationTimeOutException {

        // ===============================================================
        // Theta Parameters Array
        betaParams = new HashMap<>();
        for (String uniqueClass : uniqueLabelCount.keySet()) {
            RealVector betaVector = MatrixUtils.createRealVector(VectorOperations.ones(dimension + 1));
            betaParams.put(uniqueClass, betaVector);
        }

        // ===============================================================
        // Newton's Method for Optimization
        double delta, error0 = 0;
        int index = 0;

        for (int jdx = 0; jdx < 100; jdx++) {

            initializeTrainingData(index);

            // ===============================================================
            // Compute DelEDelE @ iteration
            LRGradientGenerator gradGenerator = new LRGradientGenerator(
                    xTilda, newSetOfTrainingPatterns, yResponse, betaParams);

            // ===============================================================
            // Generate Gradents & Change
            RealMatrix xTyMinusP = gradGenerator.getDelLdelBeta();

            RealMatrix doubleDelLDelBeta = gradGenerator.getDoubleDelLDelBeta();

            RealMatrix regularization = MatrixUtils.createRealIdentityMatrix(
                    doubleDelLDelBeta.getRowDimension()).scalarMultiply(regularizationValue);

            RealMatrix xTwXInverse = MatrixUtils.inverse(
                    doubleDelLDelBeta.add(regularization));

            RealMatrix updateBeta = xTwXInverse.multiply(xTyMinusP);

            // ===============================================================
            // Change of Beta Params
            double norm = updateBeta.getFrobeniusNorm();

            // ===============================================================
            // Update Beta Parameters
            RealMatrix betaMatrix = MatrixUtils.createRealMatrix(
                    MatrixOperations.packMatrix(updateBeta.getColumn(0),
                            dimension + 1)).transpose();

            int idx = 0;
            for (String uniqueClass : betaParams.keySet()) {
                RealVector betaVectorPrime = betaParams.get(uniqueClass);

                betaParams.put(uniqueClass, betaVectorPrime.add(
                        betaMatrix.getRowVector(idx)));
                idx++;
            }

            // ===============================================================
            // Error
            ClassificationResult classificationResult
                    = LRClassifierGenerator.execute(
                            new LRC(betaParams, uniqueLabelCount),
                            newSetOfCrossvalPatterns);

            double error1 = ClassifierPerformance.
                    estimateMisclassificationError(
                            classificationResult, setOfCrossvalClasses);

            delta = Math.abs(error0 - error1);
            error0 = error1;

            LOGGER.info("With LRC: " + error1);
            
            if (delta < 0.0002) {
                break;
            } else if (jdx == 99) {
                throw new IterationTimeOutException("Iterations Exceeded Max");
            }

        }

        return new LRC(betaParams, uniqueLabelCount);
    }

    /**
     *
     * @param index
     */
    private void initializeTrainingData(int index) {

        // ===============================================================
        // Training and Crossval
        newSetOfTrainingPatterns = new HashMap<>();
        newSetOfTrainingClasses = new HashMap<>();

        newSetOfCrossvalPatterns = new HashMap<>();
        setOfCrossvalClasses = new HashMap<>();

        for (Integer idx : trainTest.getCrossvalMap().keySet()) {

            List<Integer> training = trainTest.getCrossvalMap().get(idx);

            if (idx == index) {
                for (Integer jdx : training) {
                    newSetOfTrainingPatterns.put(jdx, setOfPatterns.get(jdx));
                    newSetOfTrainingClasses.put(jdx, setOfClasses.get(jdx));
                }
            } else {
                for (Integer jdx : training) {
                    newSetOfCrossvalPatterns.put(jdx, setOfPatterns.get(jdx));
                    setOfCrossvalClasses.put(jdx, setOfClasses.get(jdx));
                }
            }
        }

        this.number = newSetOfTrainingPatterns.size();

        // ===============================================================
        // Get Class Members
        initializeMaps(newSetOfTrainingPatterns, newSetOfTrainingClasses);

        // ================================
        // Generate xTilda    
        xTilda = MatrixUtils.createRealMatrix(
                number * (betaParams.size()), (dimension + 1) * (betaParams.size()));

        RealMatrix xBold = MatrixUtils.createRealMatrix(number, dimension + 1);

        int counter = 0;
        for (Integer idx : newSetOfTrainingPatterns.keySet()) {
            RealVector currentPattern = newSetOfTrainingPatterns.get(idx);
            xBold.setRowVector(counter, currentPattern);
            counter++;
        }

        for (int idx = 0; idx < betaParams.size(); idx++) {
            xTilda.setSubMatrix(xBold.getData(),
                    idx * (number), idx * (dimension + 1));
        }

    }

    /**
     *
     * @param setOfPatterns
     * @param setOfClasses
     */
    private void initializeMaps(Map<Integer, RealVector> setOfPatterns,
            Map<Integer, String> setOfClasses) {
        // ===============================================================
        // Get Class Members
        classMembers = LabelHandling.sortIntoMaps(
                setOfPatterns, setOfClasses);

        // ===============================================================
        // Add One to The Front of the Training Set
        newSetOfTrainingPatterns = new HashMap<>();

        for (Integer idx : setOfPatterns.keySet()) {
            RealVector withOne = MatrixUtils.createRealVector(VectorOperations.ones(
                    setOfPatterns.get(idx).getDimension() + 1));
            withOne.setSubVector(1, setOfPatterns.get(idx));

            newSetOfTrainingPatterns.put(idx, withOne);
        }

        // ===============================================================
        // Initialize Response and Beta Parameters Array
        yResponse = new HashMap<>();
        List<String> uniqueClasses = new ArrayList<>(uniqueLabelCount.keySet());
        for (Integer idx : setOfPatterns.keySet()) {
            RealVector yVector = new ArrayRealVector(classMembers.keySet().size());

            for (int jdx = 0; jdx < uniqueClasses.size(); jdx++) {
                if (setOfClasses.get(idx).equalsIgnoreCase(
                        uniqueClasses.get(jdx))) {
                    yVector.setEntry(jdx, 1);
                }
            }
            yResponse.put(idx, yVector);
        }
    }

    // ====================================================================
    /**
     *
     * @param lrc
     * @param inputPatternMap
     * <p>
     * @return
     */
    public static ClassificationResult execute(LRC lrc,
            Map<Integer, RealVector> inputPatternMap) {

        Map<Integer, String> labelEstimate = new HashMap<>();
        Map<Integer, Map<String, Double>> labelPostProb = new HashMap<>();

        // ========================================        
        // Loop Over Cross-Validation Data
        for (Integer crossvalID : inputPatternMap.keySet()) {

            RealVector pattern = inputPatternMap.get(crossvalID);
            Pair<String, Map<String, Double>> estimatedLabel
                    = generateClassEstimates(pattern, lrc);

            labelEstimate.put(crossvalID, estimatedLabel.getKey());
            labelPostProb.put(crossvalID, estimatedLabel.getValue());

        }

        return new ClassificationResult(labelEstimate,
                labelPostProb, lrc.getTrainingDataCount());
    }

    /**
     *
     * @param lrc
     * @param inputPatternMap
     * @param inputClassMap
     * @return
     */
    public static double estimateMisclassificationRate(LRC lrc,
            Map<Integer, RealVector> inputPatternMap,
            Map<Integer, String> inputClassMap) {

        int misclass = 0;
        for (Integer idx : inputClassMap.keySet()) {
            RealVector pattern = inputPatternMap.get(idx);
            Pair<String, Map<String, Double>> estimatedLabel
                    = generateClassEstimates(pattern, lrc);

            if (!estimatedLabel.getKey().equalsIgnoreCase(inputClassMap.get(idx))) {
                misclass++;
            }
        }

        return (double) misclass / (double) inputClassMap.size();
    }

    /**
     *
     * @param currentPattern
     * @param lrc
     * @return
     */
    public static Pair<String, Map<String, Double>> generateClassEstimates(
            RealVector currentPattern, LRC lrc) {

        RealVector patternUpdate = MatrixUtils.createRealVector(VectorOperations.ones(currentPattern.getDimension() + 1));
        patternUpdate.setSubVector(1, currentPattern);

        RealVector softMaxProb = SupportingFunctionality.SoftMax(
                lrc.getBetaParams(), patternUpdate);

        int idx = softMaxProb.getMaxIndex();
        String[] classTypes = lrc.getBetaParams().keySet().toArray(
                new String[lrc.getBetaParams().keySet().size()]);

        Map<String, Double> postProbMap = new HashMap<>();
        for (int jdx = 0; jdx < classTypes.length; jdx++) {
            postProbMap.put(classTypes[jdx], softMaxProb.getEntry(jdx));
        }

        if (idx == -1) {
            return new ImmutablePair<>("Missed", postProbMap);
        } else {
            return new ImmutablePair<>(classTypes[idx], postProbMap);
        }

    }

    // ====================================================================
    /**
     *
     * @param regularizationValue
     */
    public void setRegularizationValue(double regularizationValue) {
        this.regularizationValue = regularizationValue;
    }

}
