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
package fit.astro.vsa.utilities.ml.oc;

import fit.astro.vsa.common.bindings.math.kernel.GenerateKernelFunction;
import fit.astro.vsa.common.bindings.math.kernel.KernelType;
import fit.astro.vsa.common.bindings.ml.ClassificationResult;
import static java.lang.Double.NaN;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.linear.RealVector;

/**
 * Tax, D., & Duin, R. (2001). Combining one-class classifiers. Multiple
 * Classifier Systems, 299-308.
 *
 * http://jeroenjanssens.com/jeroenjanssens-thesis.pdf
 *
 * http://www.dtic.mil/dtic/tr/fulltext/u2/a281222.pdf
 *
 * Pages 13, EQuation 2.32 & Babich, Gregory A., and Octavia I. Camps. "Weighted
 * Parzen windows for pattern classification." IEEE Transactions on Pattern
 * Analysis and Machine Intelligence 18.5 (1996): 567-570.
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class OneClassPWC {

    // ============================================================
    // Input
    private final Map<Integer, RealVector> setOfTrainingData;
    private final String target;
    private final String anomaly;

    // ============================================================
    // Adjustable Parameters
    private KernelType kernelType = KernelType.GAUSSIAN;

    /**
     *
     * @param setOfTrainingData
     * @param target
     * @param anomaly
     */
    public OneClassPWC(Map<Integer, RealVector> setOfTrainingData,
            String target, String anomaly) {
        this.setOfTrainingData = setOfTrainingData;
        this.target = target;
        this.anomaly = anomaly;
    }

    /**
     * Training the OC-PWC Based on input Spread and Training Data Using
     * Leave-One-Out-Cross-Validation training requires 0% missed detections
     *
     * @param spread
     * @return
     */
    public double train(double spread) {

        UnivariateFunction kernelToUse
                = GenerateKernelFunction.generateUnivariateKernel(kernelType);

        Map<String, Integer> uniqueLabelCount = new HashMap<>();

        uniqueLabelCount.put(target, setOfTrainingData.size());

        double min = NaN;
        // =========================================================
        // Loop Over Data
        for (Entry<Integer, RealVector> entry : setOfTrainingData.entrySet()) {

            double pdf = estimatePDFUnivar(entry, spread, kernelToUse);

            if (Double.isNaN(pdf)) {
                pdf = 0.0;
            }

            if (!Double.isNaN(min)) {
                min = Math.min(min, pdf);
            } else {
                min = pdf;
            }

        }

        return min;
    }

    /**
     *
     * @param spread
     * @param inputPatternMap
     * @param threshold
     *
     * @return
     */
    public ClassificationResult execute(double spread,
            Map<Integer, RealVector> inputPatternMap, double threshold) {

        Map<Integer, String> labelEstimate = new HashMap<>();
        Map<Integer, Map<String, Double>> labelPostProb = new HashMap<>();

        UnivariateFunction kernelToUse = GenerateKernelFunction
                .generateUnivariateKernel(kernelType);

        Map<String, Integer> uniqueLabelCount = new HashMap<>();

        uniqueLabelCount.put(target, setOfTrainingData.size());

        // =========================================================
        // Loop Over Data
        for (Entry<Integer, RealVector> entry : inputPatternMap.entrySet()) {

            double pdf = estimatePDFUnivar(entry.getValue(), spread, kernelToUse);

            if (Double.isNaN(pdf)) {
                pdf = 0.0;
            }

            /**
             * Hard Probabilities for OC
             */
            Map<String, Double> postProb = new HashMap<>();
            if (pdf < threshold) {
                postProb.put(anomaly, 1.0);
                postProb.put(target, 0.0);
                labelPostProb.put(entry.getKey(), postProb);
                labelEstimate.put(entry.getKey(), anomaly);
            } else {
                postProb.put(anomaly, 0.0);
                postProb.put(target, 1.0);
                labelPostProb.put(entry.getKey(), postProb);
                labelEstimate.put(entry.getKey(), target);
            }

        }

        ClassificationResult classResult = new ClassificationResult(
                labelEstimate, labelPostProb, uniqueLabelCount);

        classResult.setThreshold(threshold);

        return classResult;
    }

    private double estimatePDFMulitvar(RealVector observedPattern, double spread,
            UnivariateFunction kernelToUse) {

        double elementSum = 0.0;
        // =========================================================
        // Cycle over training data
        for (Integer jdx : setOfTrainingData.keySet()) {

            //Find Delta Vector
            RealVector deltaIJ = observedPattern.subtract(setOfTrainingData.get(jdx));

            //Euclidean Distance
            double kernelMulti = 1;
            for (int idx = 0; idx < deltaIJ.getDimension(); idx++) {
                kernelMulti = kernelMulti
                        * kernelToUse.value(deltaIJ.getEntry(idx) / spread);
            }

            elementSum += kernelMulti / (Math.pow(spread, deltaIJ.getDimension()));
        }

        return elementSum / (setOfTrainingData.size());
    }

    /**
     *
     * @param x_i
     * @param spread
     * @param kernelToUse
     * @return
     */
    private double estimatePDFUnivar(RealVector x_i, double spread,
            UnivariateFunction kernelToUse) {

        double elementSum = 0.0;
        // =========================================================
        // Cycle over training data
        for (Integer jdx : setOfTrainingData.keySet()) {
            RealVector x_j = setOfTrainingData.get(jdx);
            //Find Delta Vector
            RealVector deltaIJ = x_i.subtract(x_j);
            //Euclidean Distance
            double xMinusX = Math.sqrt(deltaIJ.dotProduct(deltaIJ)) / spread;

            double pdf = kernelToUse.value(xMinusX);

            if (Double.isNaN(pdf)) {
                pdf = 0;
            }

            elementSum += pdf;
        }

        return elementSum / (setOfTrainingData.size() * spread);
    }

    /**
     *
     * @param entry
     * @param spread
     * @param kernelToUse
     * @return
     */
    private double estimatePDFUnivar(Entry<Integer, RealVector> entry, double spread,
            UnivariateFunction kernelToUse) {

        double elementSum = 0.0;
        // =========================================================
        // Cycle over training data
        for (Entry<Integer, RealVector> training : setOfTrainingData.entrySet()) {

            if (training.equals(entry)) {
                continue;
            }

            //Find Delta Vector
            RealVector deltaIJ = entry.getValue().subtract(
                    training.getValue());
            //Euclidean Distance
            double xMinusX = Math.sqrt(deltaIJ.dotProduct(deltaIJ)) / spread;

            double pdf = kernelToUse.value(xMinusX);

            if (Double.isNaN(pdf)) {
                pdf = 0;
            }

            elementSum += pdf;
        }

        return elementSum / (setOfTrainingData.size() * spread);

    }

    /**
     * Set the kernel type.
     *
     * @param kernelType
     */
    public void setKernelType(KernelType kernelType) {
        this.kernelType = kernelType;
    }

}
