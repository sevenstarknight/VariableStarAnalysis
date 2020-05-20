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
import fit.astro.vsa.common.bindings.ml.metric.MultiViewMetric_MV;
import fit.astro.vsa.common.datahandling.LabelHandling;
import fit.astro.vsa.utilities.ml.knn.KNNMultiMatrixMetric;
import fit.astro.vsa.utilities.ml.metriclearning.l3ml_mv.LargeMarginMultiMetricLearning_MV;
import fit.astro.vsa.utilities.ml.performance.ClassifierPerformance;
import fit.astro.vsa.utilities.ml.performance.ConfusionMatrix;
import fit.astro.vsa.common.datahandling.training.multi.TrainCrossMultiViewData_MV;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author kjohnston
 */
public class ProcessTimeDomainViaL3ML_MV {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(ProcessTimeDomainViaL3ML_MV.class);

    protected static Map<Integer, Map<String, RealMatrix>> setOfPatterns_Training;
    protected static Map<Integer, String> setOfClasses_Training;

    protected static Map<Integer, Map<String, RealMatrix>> setOfPatterns_Testing;
    protected static Map<Integer, String> setOfClasses_Testing;

    protected static Map<Integer, List<Integer>> crossvalMap;

    protected static int lmnnK = 13;

 
    protected static void trainL3ML(String location, double lambda, double mu, double gamma)
            throws IOException, NotEnoughDataException {

        Map<Integer, Map<String, RealMatrix>> mapOfPatterns = new HashMap<>();
        Map<Integer, String> mapOfClasses = new HashMap<>();

        Map<String, List<Integer>> classMembers = LabelHandling
                .sortIntoMaps(setOfClasses_Training);

        int maxSize = Integer.MAX_VALUE;
        for (String label : classMembers.keySet()) {
            if (maxSize > classMembers.get(label).size()) {
                maxSize = classMembers.get(label).size();
            }
        }

        Random rand = new Random(42L);
        for (String label : classMembers.keySet()) {

            List<Integer> list = new ArrayList<>(classMembers.get(label));
            Collections.shuffle(list, rand);

            for (int idx = 0; idx < maxSize; idx++) {
                mapOfPatterns.put(list.get(idx), setOfPatterns_Training.get(list.get(idx)));
                mapOfClasses.put(list.get(idx), setOfClasses_Training.get(list.get(idx)));
            }
        }

        //===================================================================
        // Try L3ML
        LargeMarginMultiMetricLearning_MV l3ml_mv
                = new LargeMarginMultiMetricLearning_MV(
                        mapOfPatterns, mapOfClasses);

        l3ml_mv.setLAMBDA(lambda);
        l3ml_mv.setGAMMA(gamma);
        l3ml_mv.setMU(mu);

        Map<String, MultiViewMetric_MV> outputVar = l3ml_mv.execute(lmnnK);

        // ==================================================================
        int[] kNeigh = VectorOperations.linearSpace(19, 1, 2);

        RealMatrix errorTrain = new Array2DRowRealMatrix(kNeigh.length, 3);

        for (Integer idx : crossvalMap.keySet()) {

            TrainCrossMultiViewData_MV crossDataKNN;
            try {
                crossDataKNN = new TrainCrossMultiViewData_MV(
                        setOfPatterns_Training, setOfClasses_Training, crossvalMap, idx);
            } catch (NullPointerException exception) {
                LOGGER.error(exception.getMessage());
                crossDataKNN = new TrainCrossMultiViewData_MV(
                        setOfPatterns_Training, setOfClasses_Training, crossvalMap, idx);
            }

            // =============== Train and Apply Classifiers
            KNNMultiMatrixMetric knn = new KNNMultiMatrixMetric(
                    crossDataKNN.getSetOfTrainingPatterns(),
                    crossDataKNN.getSetOfTrainingClasses());

            Map<Integer, ClassificationResult> knnWithResults = knn.execute(kNeigh,
                    "Missed", outputVar, crossDataKNN.getSetOfCrossvalPatterns());

            int counter = 0;
            RealMatrix errorMatrix = new Array2DRowRealMatrix(knnWithResults.size(), 2);
            for (Integer kValue : knnWithResults.keySet()) {

                double errorWithNow = ClassifierPerformance
                        .estimateMisclassificationError(knnWithResults.get(kValue),
                                crossDataKNN.getSetOfCrossvalClasses());

                errorMatrix.setRow(counter, new double[]{kValue, errorWithNow});
                counter++;
            }

            errorTrain = errorTrain.add(errorMatrix);
        }

        errorTrain = errorTrain.scalarMultiply((1 / (double) crossvalMap.keySet().size()));

        for (int idx = 0; idx < kNeigh.length; idx++) {
            LOGGER.info("===========================================");
            LOGGER.info("With L3ML-MV, k-NN: " + errorTrain.getEntry(idx, 0));
            LOGGER.info("With Learned Metric Error: " + errorTrain.getEntry(idx, 1));
        }

        MLArray errorML = new MLDouble("error", errorTrain.getData());
        List<MLArray> list = new ArrayList<>();
        list.add(errorML);

        MatlabFunctions.storeToFinal("Training-" + location, list);
    }

    protected static void testL3ML(int kNeig, double lambda, double mu, double gamma,
            String location) throws IOException, NotEnoughDataException {

        LOGGER.info("Lambda: " + lambda + "  Mu:" + mu + "  Gamma:" + gamma);
        Map<Integer, Map<String, RealMatrix>> mapOfPatterns = new HashMap<>();
        Map<Integer, String> mapOfClasses = new HashMap<>();

        Map<String, List<Integer>> classMembers = LabelHandling
                .sortIntoMaps(setOfClasses_Training);

        int maxSize = Integer.MAX_VALUE;
        for (String label : classMembers.keySet()) {
            if (maxSize > classMembers.get(label).size()) {
                maxSize = classMembers.get(label).size();
            }
        }

        Random rand = new Random(42L);
        for (String label : classMembers.keySet()) {

            List<Integer> list = new ArrayList<>(classMembers.get(label));
            Collections.shuffle(list, rand);

            for (int idx = 0; idx < maxSize; idx++) {
                mapOfPatterns.put(list.get(idx), setOfPatterns_Training.get(list.get(idx)));
                mapOfClasses.put(list.get(idx), setOfClasses_Training.get(list.get(idx)));
            }
        }

        //===================================================================
        // Try L3ML
        LargeMarginMultiMetricLearning_MV l3ml_mv
                = new LargeMarginMultiMetricLearning_MV(
                        mapOfPatterns, mapOfClasses);

        l3ml_mv.setLAMBDA(lambda);
        l3ml_mv.setGAMMA(gamma);
        l3ml_mv.setMU(mu);

        Map<String, MultiViewMetric_MV> outputVar = l3ml_mv.execute(lmnnK);

        // =============== Train and Apply Classifiers
        KNNMultiMatrixMetric knn = new KNNMultiMatrixMetric(
                setOfPatterns_Training,
                setOfClasses_Training);

        // Test
        ClassificationResult knnSelf = knn.execute(kNeig,
                "Missed", outputVar, setOfPatterns_Training);

        double errorSelf = ClassifierPerformance
                .estimateMisclassificationError(knnSelf,
                        setOfClasses_Training);
        
        LOGGER.info("Self Test: " + errorSelf);


        ClassificationResult knnWithResults = knn.execute(kNeig,
                "Missed", outputVar, setOfPatterns_Testing);

        double errorWithNow = ClassifierPerformance
                .estimateMisclassificationError(knnWithResults,
                        setOfClasses_Testing);


        LOGGER.info("===========================================");
        LOGGER.info("With L3ML-MV, k-NN: " + kNeig);
        LOGGER.info("With Learned Metric Error: " + errorWithNow);

        LOGGER.info("===========================================");
        ConfusionMatrix confMatrixWith = new ConfusionMatrix(setOfClasses_Testing, knnWithResults);
        LOGGER.info("With L3ML-MV, Conf Matrix With");
        confMatrixWith.printConfusionMatrix();
        
        confMatrixWith.printCountMatrix();

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
        list.add(countsMLWith);
        list.add(labelsML);

        MatlabFunctions.storeToFinal("Error-" + location, list);

    }

}
