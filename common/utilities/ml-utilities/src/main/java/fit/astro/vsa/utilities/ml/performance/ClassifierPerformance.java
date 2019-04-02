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
package fit.astro.vsa.utilities.ml.performance;

import fit.astro.vsa.common.bindings.ml.ClassificationResult;
import java.util.Map;

/**
 * Fawcett, T. (2006). An introduction to ROC analysis. Pattern recognition
 * letters, 27(8), 861-874.
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class ClassifierPerformance {

    private final double truePositive;
    private final double falseNegative;
    private final double trueNegative;
    private final double falsePositive;

    private final double total;

    /**
     * 
     * @param truePositive
     * @param falseNegative
     * @param trueNegative
     * @param falsePositive 
     */
    public ClassifierPerformance(double truePositive, double falseNegative, 
            double trueNegative, double falsePositive) {
        this.truePositive = truePositive;
        this.falseNegative = falseNegative;
        this.trueNegative = trueNegative;
        this.falsePositive = falsePositive;

        this.total = truePositive + falseNegative
                + trueNegative + falsePositive;
    }

    /**
     * (trueNegative + truePositive) / total
     *
     * @return
     */
    public double getAccuracy() {
        return (trueNegative + truePositive) / total;
    }

    /**
     * (truePositive) / (truePositive + falsePositive)
     *
     * @return
     */
    public double getPrecision() {
        return (truePositive) / (truePositive + falsePositive);
    }

    /**
     * (falsePositive) / (falsePositive + trueNegative)
     *
     * @return
     */
    public double getFPRate() {
        return (falsePositive) / (falsePositive + trueNegative);
    }

    /**
     * (truePositive) / (truePositive + falseNegative)
     *
     * @return
     */
    public double getRecall() {
        return (truePositive) / (truePositive + falseNegative);
    }

    /**
     * Estimate the misclassification rate (Wrong/Total)
     *
     * @param classificationResult
     * @param setOfCrossClasses
     * @return
     */
    public static double estimateMisclassificationError(
            ClassificationResult classificationResult,
            Map<Integer, String> setOfCrossClasses) {

        double error = 0;
        error = classificationResult.getLabelEstimate()
                .keySet().stream().filter((kdx)
                        -> (!classificationResult.getLabelEstimate().get(kdx)
                        .equalsIgnoreCase(setOfCrossClasses.get(kdx))))
                .map((_item) -> 1.0)
                .reduce(error, (accumulator, _item) -> accumulator + 1);

        return error / (double) classificationResult.getLabelEstimate().keySet().size();

    }

    /**
     *
     * @param metric
     * @return [metric, TP, FN, TN, FP, Accuracy, Precision, FPR, Recall]
     */
    public double[] estimatePerformanceSet(double metric) {

        return new double[]{metric,
            truePositive, falseNegative, trueNegative, falsePositive,
            getAccuracy(), getPrecision(), getFPRate(), getRecall()};

    }

}
