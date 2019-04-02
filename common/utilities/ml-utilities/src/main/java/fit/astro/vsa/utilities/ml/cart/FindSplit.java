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
package fit.astro.vsa.utilities.ml.cart;

import fit.astro.vsa.common.utilities.math.support.SortingOperations;
import fit.astro.vsa.common.datahandling.LabelHandling;
import static fit.astro.vsa.common.utilities.math.NumericalConstants.LOG2;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.math3.linear.RealVector;

/**
 * Find the split in the parent node, which maximizes the purity in the
 * individual nodes
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class FindSplit {

    /**
     * Generate the split in the dataset based on an impurity type
     *
     * @param impurityType (impurity measure)
     * @param setOfData (set of data from parent)
     * @param setOfClasses (set of classes from parent)
     * @param dimensions (size of the pattern)
     * @return
     */
    public static ImmutablePair<Integer, Double> GenerateSplit(
            ImpurityType impurityType,
            Map<Integer, RealVector> setOfData,
            Map<Integer, String> setOfClasses, int dimensions) {

        // Estimate the impurity of the parent
        double parentImpurity = EstimateImpurity(impurityType, setOfClasses);

        Map<Integer, Double> thresholdMap = new HashMap<>();
        Map<Integer, Double> deltaEntropyMap = new HashMap<>();

        // Scan over dimensions
        for (int idx = 0; idx < dimensions; idx++) {

            // Get the value, at the dimension, for each pattern
            List<Double> valueTmp = new ArrayList<>();
            for (Integer jdx : setOfData.keySet()) {
                RealVector currentPattern = setOfData.get(jdx);
                valueTmp.add(currentPattern.getEntry(idx));
            }

            // Sort the values
            Collections.sort(valueTmp);

            double deltaMaxEntropy = 0;
            double threshold;
            // Use the value as a split, estimate impurity of children
            for (Double e : valueTmp) {

                double childImpurity = EstimateImpurityOfChildren(impurityType,
                        setOfData, setOfClasses, e, idx);

                // Is the delta better then the max so far?
                double deltaEntropy = parentImpurity - childImpurity;

                if (deltaEntropy > deltaMaxEntropy) {
                    deltaMaxEntropy = deltaEntropy;
                    threshold = e;
                    //
                    thresholdMap.put(idx, threshold);
                    deltaEntropyMap.put(idx, deltaMaxEntropy);
                }
            }

        }

        if (thresholdMap.isEmpty()) {
            // No threshold, no split
            return new ImmutablePair<>(null, null);
        } else {
            // Go and generate 
            Map<Integer, Double> sortedEntropy
                    = SortingOperations.sortByDecendingValue(
                            deltaEntropyMap);

            Integer selectedDimension = sortedEntropy.keySet()
                    .iterator().next();
            Double selectedThreshold = thresholdMap
                    .get(selectedDimension);

            return new ImmutablePair<>(
                    selectedDimension, selectedThreshold);
        }
    }

    /**
     * Estimate the impurity of the children based on
     *
     * @param impurityType
     * @param setOfData
     * @param setOfClasses
     * @param threshold
     * @param idxDimension
     * @return
     */
    private static double EstimateImpurityOfChildren(ImpurityType impurityType,
            Map<Integer, RealVector> setOfData,
            Map<Integer, String> setOfClasses,
            double threshold, int idxDimension) {

        // Store L/R
        Map<Integer, String> setOfLeftClasses = new HashMap<>();
        Map<Integer, String> setOfRightClasses = new HashMap<>();

        // Split the Parent
        for (Integer idx : setOfData.keySet()) {

            if (setOfData.get(idx).getEntry(idxDimension) > threshold) {
                setOfLeftClasses.put(idx, setOfClasses.get(idx));
            } else {
                setOfRightClasses.put(idx, setOfClasses.get(idx));
            }
        }

        // Estimate Left, Estimate Right
        double gdiLeft = EstimateImpurity(impurityType, setOfLeftClasses);
        double gdiRight = EstimateImpurity(impurityType, setOfRightClasses);

        // Generate Post Impurity
        double postImpurity
                = ((double) setOfLeftClasses.size() / (double) setOfData.size()) * gdiLeft
                + ((double) setOfRightClasses.size() / (double) setOfData.size()) * gdiRight;

        return postImpurity;
    }

    /**
     * Estimate impurity based on the input class set
     *
     * @param impurityType
     * @param setOfClasses
     * @return
     */
    public static double EstimateImpurity(ImpurityType impurityType,
            Map<Integer, String> setOfClasses) {

        Map<String, Integer> uniqueClasses
                = LabelHandling.countUniqueClasses(setOfClasses);

        switch (impurityType) {

            case MISCLASSIFICATION:
                //Misclassification Reduction
                double maxMiss = 0;
                for (String e : uniqueClasses.keySet()) {
                    double p_i = uniqueClasses.get(e) / (double) setOfClasses.size();
                    if (p_i > maxMiss) {
                        maxMiss = p_i;
                    }
                }

                return 1 - maxMiss;

            case ENTROPY:
                // Information Gain (Entropy)
                double entropy = 0;
                for (String e : uniqueClasses.keySet()) {
                    double postProb = (double) uniqueClasses.get(e) / (double) setOfClasses.size();
                    entropy = entropy + postProb * Math.log(postProb) / LOG2;
                }

                return -entropy;

            case GDI:
                //Gini Diversity Index
                double gdi = 0;
                for (String e : uniqueClasses.keySet()) {
                    double fraction = (double) uniqueClasses.get(e) / (double) setOfClasses.size();
                    gdi = gdi + fraction * (1 - fraction);
                }

                return gdi;

            default:
                throw new ArithmeticException("Do not know impurity type");
        }
    }
}
