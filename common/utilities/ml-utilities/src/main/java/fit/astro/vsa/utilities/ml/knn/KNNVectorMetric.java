/*
 * Copyright (C) 2016 Kyle Johnston
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without isEven the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package fit.astro.vsa.utilities.ml.knn;

import fit.astro.vsa.common.utilities.math.NumericTests;
import fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException;
import fit.astro.vsa.common.utilities.math.support.SortingOperations;
import fit.astro.vsa.common.bindings.ml.ClassificationResult;
import fit.astro.vsa.utilities.ml.MetricDistance;
import fit.astro.vsa.common.datahandling.LabelHandling;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * Keller, J. M., Gray, M. R., & Givens, J. A. (1985). A fuzzy k-nearest
 * neighbor algorithm. IEEE transactions on systems, man, and cybernetics, (4),
 * 580-585.
 *
 * Weinberger, K. Q., Blitzer, J., & Saul, L. K. (2005). Distance metric
 * learning for large margin nearest neighbor classification. In Advances in
 * neural information processing systems (pp. 1473-1480).
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class KNNVectorMetric {

    // ============================================
    // Input
    private final Map<Integer, RealVector> setOfTrainingData;
    private final Map<Integer, String> setOfTrainingClasses;

    private boolean isEuclidean = Boolean.FALSE;

    /**
     *
     * @param setOfTrainingData the set of training data which is n x p
     * @param setOfTrainingClasses the set of training labels which are 1 x p
     * @throws NotEnoughDataException
     */
    public KNNVectorMetric(
            Map<Integer, RealVector> setOfTrainingData,
            Map<Integer, String> setOfTrainingClasses) throws NotEnoughDataException {
        this.setOfTrainingData = setOfTrainingData;
        this.setOfTrainingClasses = setOfTrainingClasses;

        if (setOfTrainingData == null
                || setOfTrainingData.values() == null
                || setOfTrainingData.size() < 1) {
            throw new NotEnoughDataException("No data given, can not make classifier");
        }
    }

    /**
     *
     * @param kValue
     * @param missedLabel
     * @param inputPatternMap
     *
     * @return
     */
    public ClassificationResult execute(int kValue,
            String missedLabel, Map<Integer, RealVector> inputPatternMap) {

        RealVector firstVector
                = setOfTrainingData.values().iterator().next();

        isEuclidean = Boolean.TRUE;
        RealMatrix metricMatrix = MatrixUtils.createRealIdentityMatrix(
                firstVector.getDimension());

        return execute(kValue, missedLabel, metricMatrix, inputPatternMap);
    }

    /**
     *
     * @param kValue
     * @param missedLabel
     * @param metricMatrix
     * @param inputPatternMap
     *
     * @return
     */
    public ClassificationResult execute(int kValue, String missedLabel,
            RealMatrix metricMatrix, Map<Integer, RealVector> inputPatternMap) {

        MetricDistance metricDistance = new MetricDistance(metricMatrix);
        Map<Integer, String> labelEstimate = new HashMap<>();
        Map<Integer, Map<String, Double>> labelPostProb = new HashMap<>();

        Map<String, Integer> uniqueLabelCount
                = LabelHandling.countUniqueClasses(setOfTrainingClasses);
        // ========================================        
        // Loop Over Input Data
        for (Integer idx : inputPatternMap.keySet()) {

            RealVector crossvalPattern = inputPatternMap.get(idx);

            // ==============================================================
            // Generate Distances
            Map<Integer, Double> setOfDistances = new HashMap<>();

            for (Integer trainingDataID : setOfTrainingData.keySet()) {

                Double distance;
                if (isEuclidean) {
                    distance = setOfTrainingData.get(trainingDataID).getDistance(crossvalPattern);
                } else {
                    distance = metricDistance.distance(
                            setOfTrainingData.get(trainingDataID),
                            crossvalPattern);
                }

                setOfDistances.put(trainingDataID, distance);
            }

            // ==============================================================
            // Find Nearest Neighbors
            Map<Integer, Double> sortedDistances
                    = SortingOperations.sortByAcendingValue(setOfDistances);

            // Post Prob Estimates for Class
            Map<String, Double> nearestNeighborResponse = new HashMap<>();

            uniqueLabelCount.keySet().stream().forEach((uniqueLabels) -> {
                nearestNeighborResponse.put(uniqueLabels, 0.0);
            });

            // k nearest
            int counter = 0;
            for (Integer jdx : sortedDistances.keySet()) {
                if (nearestNeighborResponse.containsKey(setOfTrainingClasses.get(jdx))) {
                    String classLabel = setOfTrainingClasses.get(jdx);
                    Double currentCount = nearestNeighborResponse.get(classLabel);
                    nearestNeighborResponse.put(classLabel,
                            currentCount + 1.0 / (double) kValue);
                    counter++;
                }

                if (counter == kValue) {
                    break;
                }
            }

            labelPostProb.put(idx, nearestNeighborResponse);

            // ==============================================================
            // Determine Class Based on Sorted Post Prob           
            Map<String, Double> sortedNeighbors
                    = SortingOperations.sortByDecendingValue(nearestNeighborResponse);

            List<Entry<String, Double>> knnList = new ArrayList<>(
                    sortedNeighbors.entrySet());

            if (NumericTests.isApproxEqual(
                    knnList.get(0).getValue(),
                    knnList.get(1).getValue())) {
                // tie is "missed"
                labelEstimate.put(idx, missedLabel);
            } else {
                // winner take all
                labelEstimate.put(idx, knnList.get(0).getKey());
            }

        }

        return new ClassificationResult(labelEstimate, labelPostProb, uniqueLabelCount);
    }

    /**
     *
     * @param kValues
     * @param missedLabel
     * @param inputPatternMap
     *
     * @return
     */
    public Map<Integer, ClassificationResult> execute(int[] kValues,
            String missedLabel, Map<Integer, RealVector> inputPatternMap) {

        RealVector firstVector
                = setOfTrainingData.values().iterator().next();

        RealMatrix metricMatrix = MatrixUtils.createRealIdentityMatrix(
                firstVector.getDimension());

        return execute(kValues, missedLabel, metricMatrix, inputPatternMap);
    }

    /**
     *
     * @param kValues
     * @param missedLabel
     * @param metricMatrix
     * @param inputPatternMap
     *
     * @return
     */
    public Map<Integer, ClassificationResult> execute(int[] kValues, String missedLabel,
            RealMatrix metricMatrix, Map<Integer, RealVector> inputPatternMap) {

        MetricDistance metricDistance = new MetricDistance(metricMatrix);

        Map<String, Integer> uniqueLabelCount
                = LabelHandling.countUniqueClasses(setOfTrainingClasses);

        Map<Integer, ClassificationResult> results = new HashMap<>();

        for (int idx = 0; idx < kValues.length; idx++) {
            Map<Integer, String> labelEstimate = new HashMap<>();
            Map<Integer, Map<String, Double>> labelPostProb = new HashMap<>();

            results.put(kValues[idx], new ClassificationResult(labelEstimate, labelPostProb, uniqueLabelCount));
        }

        // ========================================        
        // Loop Over Input Data
        for (Integer idx : inputPatternMap.keySet()) {

            RealVector crossvalPattern = inputPatternMap.get(idx);

            // ==============================================================
            // Generate Distances
            Map<Integer, Double> setOfDistances = new HashMap<>();

            for (Integer trainingDataID : setOfTrainingData.keySet()) {

                Double distance;
                if (isEuclidean) {
                    distance = setOfTrainingData.get(trainingDataID).getDistance(crossvalPattern);
                } else {
                    distance = metricDistance.distance(
                            setOfTrainingData.get(trainingDataID),
                            crossvalPattern);
                }

                setOfDistances.put(trainingDataID, distance);
            }

            // ==============================================================
            // Find Nearest Neighbors
            Map<Integer, Double> sortedDistances
                    = SortingOperations.sortByAcendingValue(setOfDistances);

            // Post Prob Estimates for Class
            Map<String, Double> nearestNeighborResponse = new HashMap<>();

            uniqueLabelCount.keySet().stream().forEach((uniqueLabels) -> {
                nearestNeighborResponse.put(uniqueLabels, 0.0);
            });

            // ================================================
            for (int kdx = 0; kdx < kValues.length; kdx++) {

                int kValue = kValues[kdx];

                // k nearest
                int counter = 0;
                for (Integer jdx : sortedDistances.keySet()) {
                    if (nearestNeighborResponse.containsKey(setOfTrainingClasses.get(jdx))) {
                        String classLabel = setOfTrainingClasses.get(jdx);
                        Double currentCount = nearestNeighborResponse.get(classLabel);
                        nearestNeighborResponse.put(classLabel,
                                currentCount + 1.0 / (double) kValue);
                        counter++;
                    }

                    if (counter == kValue) {
                        break;
                    }
                }

                results.get(kValue).getLabelAndPostProb().put(idx, nearestNeighborResponse);

                // ==============================================================
                // Determine Class Based on Sorted Post Prob           
                Map<String, Double> sortedNeighbors
                        = SortingOperations.sortByDecendingValue(nearestNeighborResponse);

                List<Entry<String, Double>> knnList = new ArrayList<>(
                        sortedNeighbors.entrySet());

                if (NumericTests.isApproxEqual(
                        knnList.get(0).getValue(),
                        knnList.get(1).getValue())) {
                    // tie is "missed"
                    results.get(kValue).getLabelEstimate().put(idx, missedLabel);
                } else {
                    // winner take all
                    results.get(kValue).getLabelEstimate().put(idx, knnList.get(0).getKey());
                }
            }

        }

        return results;
    }
}
