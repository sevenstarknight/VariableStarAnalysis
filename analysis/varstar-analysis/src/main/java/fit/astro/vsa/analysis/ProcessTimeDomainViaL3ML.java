/*
 * Copyright (C) 2018 kjohnston
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
package fit.astro.vsa.analysis;

import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;
import fit.astro.vsa.common.utilities.io.MatlabFunctions;
import fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException;
import fit.astro.vsa.common.utilities.math.linearalgebra.VectorOperations;
import fit.astro.vsa.common.bindings.ml.ClassificationResult;
import fit.astro.vsa.common.bindings.math.ml.metric.MultiViewMetric;
import fit.astro.vsa.utilities.ml.knn.KNNMultiVectorMetric;
import fit.astro.vsa.utilities.ml.metriclearning.l3ml.LargeMarginMultiMetricLearning;
import fit.astro.vsa.utilities.ml.performance.ClassifierPerformance;
import fit.astro.vsa.utilities.ml.performance.ConfusionMatrix;
import fit.astro.vsa.common.datahandling.training.multi.TrainCrossMultiViewData;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author kjohnston
 */
public class ProcessTimeDomainViaL3ML {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(ProcessTimeDomainViaL3ML.class);

    protected static Map<Integer, Map<String, RealVector>> setOfPatterns_Training;
    protected static Map<Integer, String> setOfClasses_Training;

    protected static Map<Integer, Map<String, RealVector>> setOfPatterns_Testing;
    protected static Map<Integer, String> setOfClasses_Testing;

    protected static Map<Integer, List<Integer>> crossvalMap;

    protected static void trainL3ML(String location, double tau, double mu, double lambda)
            throws IOException, NotEnoughDataException {

        LOGGER.info("Tau: " + tau + "  Mu:" + mu+ "  Lambda:" + lambda);
        
        TrainCrossMultiViewData crossDataL3ML = new TrainCrossMultiViewData(
                setOfPatterns_Training, setOfClasses_Training, crossvalMap, 0);

        //===================================================================
        // Try L3ML
        LargeMarginMultiMetricLearning l3ml
                = new LargeMarginMultiMetricLearning(
                        crossDataL3ML.getSetOfTrainingPatterns(),
                        crossDataL3ML.getSetOfTrainingClasses());
        l3ml.setLAMBDA(lambda);

        Map<String, MultiViewMetric> outputVar = l3ml.execute(tau, mu);

        // ==================================================================
        int[] kNeigh = VectorOperations.linearSpace(19, 1, 2);

        RealMatrix errorTrain = new Array2DRowRealMatrix(kNeigh.length, 3);

        for (Integer idx : crossvalMap.keySet()) {

            TrainCrossMultiViewData crossDataKNN = new TrainCrossMultiViewData(
                    setOfPatterns_Training, setOfClasses_Training, crossvalMap, idx);

            // =============== Train and Apply Classifiers
            KNNMultiVectorMetric knn = new KNNMultiVectorMetric(
                    crossDataKNN.getSetOfTrainingPatterns(),
                    crossDataKNN.getSetOfTrainingClasses());

            Map<Integer, ClassificationResult> knnWithResults = knn.execute(kNeigh,
                    "Missed", outputVar, crossDataKNN.getSetOfCrossvalPatterns());
            
            Map<Integer, ClassificationResult> knnWithoutResults = knn.execute(kNeigh,
                    "Missed", crossDataKNN.getSetOfCrossvalPatterns());

            int counter = 0;
            RealMatrix errorMatrix = new Array2DRowRealMatrix(knnWithoutResults.size(), 3);
            for (Integer kValue : knnWithoutResults.keySet()) {

                double errorWithoutNow = ClassifierPerformance
                        .estimateMisclassificationError(knnWithoutResults.get(kValue),
                                crossDataKNN.getSetOfCrossvalClasses());

                double errorWithNow = ClassifierPerformance
                        .estimateMisclassificationError(knnWithResults.get(kValue),
                                crossDataKNN.getSetOfCrossvalClasses());

                errorMatrix.setRow(counter, new double[]{kValue, errorWithoutNow, errorWithNow});
                counter++;
            }

            errorTrain = errorTrain.add(errorMatrix);
        }

        errorTrain = errorTrain.scalarMultiply((1 / (double) crossvalMap.keySet().size()));

        for (int idx = 0; idx < kNeigh.length; idx++) {
            LOGGER.info("===========================================");
            LOGGER.info("k-Value: " + errorTrain.getEntry(idx, 0));
            LOGGER.info("Without Learned Metric Error: " + errorTrain.getEntry(idx, 1));
            LOGGER.info("With Learned Metric Error: " + errorTrain.getEntry(idx, 2));
        }

        MLArray errorML = new MLDouble("error", errorTrain.getData());
        List<MLArray> list = new ArrayList<>();
        list.add(errorML);

        MatlabFunctions.storeToFinal("Training-" + location, list);
    }
    
    protected static void testL3ML(int kNeig, double tau, double mu, double lambda,
            String location) throws IOException, NotEnoughDataException {

        LOGGER.info("Tau: " + tau + "  Mu:" + mu+ "  Lambda:" + lambda);
        
        //===================================================================
        // Try L3ML
        LargeMarginMultiMetricLearning l3ml
                = new LargeMarginMultiMetricLearning(
                        setOfPatterns_Training,
                        setOfClasses_Training);
        l3ml.setLAMBDA(lambda);

        Map<String, MultiViewMetric> outputVar = l3ml.execute(tau, mu);
        for (String feature : outputVar.keySet()) {
            LOGGER.info("For feature: " + feature
                    + "  , Weight: " + outputVar.get(feature).getWeight());
        }

        // =============== Train and Apply Classifiers
        KNNMultiVectorMetric knn = new KNNMultiVectorMetric(
                setOfPatterns_Training,
                setOfClasses_Training);


        ClassificationResult knnWithResults = knn.execute(kNeig,
                "Missed", outputVar, setOfPatterns_Testing);

        ClassificationResult knnWithoutResults = knn.execute(kNeig,
                "Missed", setOfPatterns_Testing);
        
        double errorWithNow = ClassifierPerformance
                .estimateMisclassificationError(knnWithResults,
                        setOfClasses_Testing);

        double errorWithoutNow = ClassifierPerformance
                .estimateMisclassificationError(knnWithoutResults,
                        setOfClasses_Testing);

        LOGGER.info("===========================================");
        LOGGER.info("With L3ML, k-NN: " + kNeig);
        LOGGER.info("With Learned Metric Error: " + errorWithNow);
        LOGGER.info("Without Learned Metric Error: " + errorWithoutNow);

        LOGGER.info("===========================================");
        LOGGER.info("With L3ML, Conf Matrix Without");
        ConfusionMatrix confMatrixWithout = new ConfusionMatrix(setOfClasses_Testing, knnWithoutResults);
        confMatrixWithout.printConfusionMatrix();

        LOGGER.info("===========================================");
        ConfusionMatrix confMatrixWith = new ConfusionMatrix(setOfClasses_Testing, knnWithResults);
        LOGGER.info("With L3ML, Conf Matrix With");
        confMatrixWith.printConfusionMatrix();

        MLArray errorMLWithout = new MLDouble("confMatrix_Without", confMatrixWithout.getConfusionMatrix().getData());
        MLArray errorMLWith = new MLDouble("confMatrix_With", confMatrixWith.getConfusionMatrix().getData());
        MLArray countsMLWith = new MLDouble("countMatrix_With", confMatrixWith.getCountMatrix().getData());
        
        List<String> listLabels = confMatrixWith.getClassTypes();

        MLCell labelsML = new MLCell("labels", new int[]{listLabels.size(), 1});

        int counter = 0;
        for (String label : listLabels) {
            MLChar labelChar = new MLChar("label", label);
            labelsML.set(labelChar, counter);
            counter++;
        }

        List<MLArray> list = new ArrayList<>();
        list.add(errorMLWith);
        list.add(errorMLWithout);
        list.add(countsMLWith);
        list.add(labelsML);

        MatlabFunctions.storeToFinal("Testing-" + location, list);

    }

}
