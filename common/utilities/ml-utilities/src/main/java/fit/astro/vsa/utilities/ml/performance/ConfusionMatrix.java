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

import fit.astro.vsa.common.utilities.math.linearalgebra.VectorOperations;
import fit.astro.vsa.common.bindings.ml.ClassificationResult;
import fit.astro.vsa.common.datahandling.LabelHandling;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class ConfusionMatrix {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(ConfusionMatrix.class);

    // ============================================
    // Input
    private final Map<Integer, String> setOfResults;
    private final Map<Integer, String> setOfInputClasses;

    // ============================================
    // Internal
    private List<String> classTypes;
    private RealMatrix confusionMatrix;
    private RealMatrix countMatrix;

    /**
     * Based on results, compare to "real values"
     *
     * @param setOfInputClasses
     * @param setOfResults
     */
    public ConfusionMatrix(Map<Integer, String> setOfInputClasses,
            Map<Integer, String> setOfResults) {
        this.setOfResults = setOfResults;
        this.setOfInputClasses = setOfInputClasses;

        generate();

    }

    /**
     * Based on results (classification results), compare to "real values"
     *
     * @param setOfInputClasses
     * @param classificationResult
     */
    public ConfusionMatrix(Map<Integer, String> setOfInputClasses,
            ClassificationResult classificationResult) {
        this.setOfResults = classificationResult.getLabelEstimate();
        this.setOfInputClasses = setOfInputClasses;

        generate();

    }

    /**
     *
     */
    private void generate() {
        Map<String, Integer> classNumbers
                = LabelHandling.countUniqueClasses(setOfInputClasses);

        classTypes = new ArrayList<>(classNumbers.keySet());
        // +1 for no clear winner
        confusionMatrix = MatrixUtils.createRealMatrix(
                classTypes.size(), classTypes.size() + 1);

        for (Integer jdx : setOfInputClasses.keySet()) {

            int idx = -1, kdx = -1;
            for (String classType : classTypes) {
                if (classType.equalsIgnoreCase(setOfInputClasses.get(jdx))) {
                    idx = classTypes.indexOf(classType);
                }

                if (classType.equalsIgnoreCase(setOfResults.get(jdx))) {
                    kdx = classTypes.indexOf(classType);
                }
            }

            if (kdx == -1) {
                //missed
                kdx = classTypes.size();
            }

            confusionMatrix.setEntry(idx, kdx,
                    confusionMatrix.getEntry(idx, kdx) + 1);

        }

        countMatrix = new Array2DRowRealMatrix(confusionMatrix.getData());

        for (int idx = 0; idx < confusionMatrix.getRowDimension(); idx++) {
            double total = VectorOperations.summationOfElements(confusionMatrix.getRowVector(idx));

            if (total != 0) {
                confusionMatrix.setRowVector(idx,
                        confusionMatrix.getRowVector(idx).mapDivide(total));
            }
        }
    }

    public double generateFScore() {

        double[] columnTotals = new double[countMatrix.getColumnDimension()];

        for (int idx = 0; idx < countMatrix.getColumnDimension(); idx++) {
            columnTotals[idx] = VectorOperations.summationOfElements(
                    countMatrix.getColumnVector(idx));
        }

        double[] rowTotals = new double[countMatrix.getRowDimension()];

        for (int idx = 0; idx < countMatrix.getRowDimension(); idx++) {
            rowTotals[idx] = VectorOperations.summationOfElements(
                    countMatrix.getRowVector(idx));
        }

        RealVector recallEstimates = new ArrayRealVector(
                countMatrix.getRowDimension());
        RealVector precisionEstimates = new ArrayRealVector(
                countMatrix.getRowDimension());
        
        for (int idx = 0; idx < countMatrix.getRowDimension(); idx++) {

            recallEstimates.setEntry(idx,
                    countMatrix.getEntry(idx, idx) / rowTotals[idx]);

            precisionEstimates.setEntry(idx,
                    countMatrix.getEntry(idx, idx) / columnTotals[idx]);
            
        }

        double meanRecall = VectorOperations.mean(recallEstimates);
        double meanPrecision = VectorOperations.mean(precisionEstimates);
     
        
        return (2*meanRecall*meanPrecision)/(meanPrecision + meanRecall);
    }

        /**
     * Print the confusion matrix
     */
    public void printCountMatrix() {

        DecimalFormat numberFormat = new DecimalFormat("#.000");

        LOGGER.info("     " + Arrays.toString(classTypes.toArray()));

        for (String label : classTypes) {
            int idx = classTypes.indexOf(label);

            double[] array = countMatrix.getRow(idx);
            String report = " ";
            for (int jndx = 0; jndx < array.length; jndx++) {
                report = report.concat(numberFormat.format(array[jndx]));
                report = report.concat("  ");
            }

            System.out.println(label + report);

        }

    }

    
    /**
     * Print the confusion matrix
     */
    public void printConfusionMatrix() {

        DecimalFormat numberFormat = new DecimalFormat("#.000");

        LOGGER.info("     " + Arrays.toString(classTypes.toArray()));

        for (String label : classTypes) {
            int idx = classTypes.indexOf(label);

            double[] array = confusionMatrix.getRow(idx);
            String report = " ";
            for (int jndx = 0; jndx < array.length; jndx++) {
                report = report.concat(numberFormat.format(array[jndx]));
                report = report.concat("  ");
            }

            System.out.println(label + report);

        }

    }

    /**
     * @return the classTypes
     */
    public List<String> getClassTypes() {
        return classTypes;
    }

    /**
     * @return the confusionMatrix
     */
    public RealMatrix getConfusionMatrix() {
        return confusionMatrix;
    }

    /**
     *
     * @return
     */
    public RealMatrix getCountMatrix() {
        return countMatrix;
    }

}
