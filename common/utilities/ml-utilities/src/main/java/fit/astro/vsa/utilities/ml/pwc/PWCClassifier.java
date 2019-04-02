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
package fit.astro.vsa.utilities.ml.pwc;

import fit.astro.vsa.common.bindings.math.kernel.GenerateKernelFunction;
import fit.astro.vsa.common.bindings.math.kernel.KernelType;
import fit.astro.vsa.common.utilities.math.linearalgebra.VectorOperations;
import fit.astro.vsa.common.bindings.ml.ClassificationResult;
import fit.astro.vsa.common.datahandling.LabelHandling;
import java.util.HashMap;
import java.util.Map;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.linear.RealVector;

/**
 * Babich, G. A., & Camps, O. I. (1996). Weighted Parzen windows for pattern
 * classification. IEEE Transactions on Pattern Analysis & Machine Intelligence,
 * (5), 567-570.
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class PWCClassifier {

    private final Map<Integer, RealVector> setOfTrainingData;
    private final Map<Integer, String> setOfTrainingClasses;

    public PWCClassifier(Map<Integer, RealVector> setOfTrainingData, Map<Integer, String> setOfTrainingClasses) {
        this.setOfTrainingData = setOfTrainingData;
        this.setOfTrainingClasses = setOfTrainingClasses;
    }

    /**
     *
     * @param kernelType
     * @param spread
     * @param missedLabel
     * @param inputPatternMap
     * <p>
     * @return
     */
    public ClassificationResult execute(KernelType kernelType, double spread,
            String missedLabel, Map<Integer, RealVector> inputPatternMap) {

        double dimensions = (double) setOfTrainingData.values().iterator().next().getDimension();

        Map<Integer, String> labelEstimate = new HashMap<>();
        Map<Integer, Map<String, Double>> labelPostProb = new HashMap<>();
        // ========================================
        // Set up: Split Input Patterns Into Class Groups

        Map<String, Integer> uniqueLabelCount
                = LabelHandling.countUniqueClasses(setOfTrainingClasses);

        Map<String, Map<Integer, RealVector>> classMembers
                = LabelHandling.sortIntoMaps(setOfTrainingData, setOfTrainingClasses);

        // initialize Prior Probabilities
        Map<String, Double> priorProbability = new HashMap<>();
        for (String e : classMembers.keySet()) {
            double sizeOfList = (double) classMembers.get(e).size();
            double priorProb = sizeOfList * Math.pow(spread, dimensions)
                    / setOfTrainingClasses.size();
            priorProbability.put(e, priorProb);
        }

        UnivariateFunction kernelToUse
                = GenerateKernelFunction.generateUnivariateKernel(kernelType);

        // Loop Over Input Data
        for (Integer idx : inputPatternMap.keySet()) {

            Map<String, Double> kernelEstimate = new HashMap<>();

            // Cycle over training data
            for (Integer jdx : setOfTrainingData.keySet()) {

                //Find Delta Vector
                RealVector deltaIJ
                        = (inputPatternMap.get(idx).subtract(
                                setOfTrainingData.get(jdx))).mapDivide(spread);

                RealVector transformedDelta = deltaIJ.map(kernelToUse);
                double productElements = VectorOperations.productOfElements(transformedDelta);

                // Store Post Prob
                String classType = setOfTrainingClasses.get(jdx);

                if (kernelEstimate.containsKey(classType)) {
                    double currentValue = kernelEstimate.get(classType);
                    kernelEstimate.put(classType, currentValue + productElements);
                } else {
                    kernelEstimate.put(classType, productElements);
                }

            }

            // Determine the Response for Each Class
            Map<String, Double> responceEstimate = new HashMap<>();
            double total = 0;
            for (String e : classMembers.keySet()) {
                double priorProb = priorProbability.get(e);
                double kernelEstimateVal = kernelEstimate.get(e);
                double response = priorProb * kernelEstimateVal;
                responceEstimate.put(e, response);
                total += response;
            }

            labelPostProb.put(idx, responceEstimate);

            // Generate The Estimate 
            double max = 0;
            String classType = null;
            if (total != 0) {
                for (String e : classMembers.keySet()) {
                    if (max < responceEstimate.get(e)) {
                        max = responceEstimate.get(e);
                        classType = e;
                    }
                }
                labelEstimate.put(idx, classType);

            } else {
                labelEstimate.put(idx, missedLabel);
            }

        }

        return new ClassificationResult(labelEstimate, labelPostProb, uniqueLabelCount);
    }

}
