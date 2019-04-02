/*
 * Copyright (C) 2019 kjohnston
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
import fit.astro.vsa.common.bindings.ml.ClassificationResult;
import fit.astro.vsa.common.datahandling.training.multi.TrainCrossMultiViewData_MV;
import fit.astro.vsa.common.utilities.io.MatlabFunctions;
import fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException;
import fit.astro.vsa.common.utilities.math.linearalgebra.VectorOperations;
import fit.astro.vsa.utilities.ml.knn.KNNMultiMatrixMetric;
import fit.astro.vsa.utilities.ml.performance.ClassifierPerformance;
import fit.astro.vsa.utilities.ml.performance.ConfusionMatrix;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author kjohnston
 */
public class ProcessRawMultiViewKNN_MV {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(ProcessRawMultiViewKNN_MV.class);
    
    protected static Map<Integer, Map<String, RealMatrix>> setOfPatterns_Training;
    protected static Map<Integer, String> setOfClasses_Training;

    protected static Map<Integer, Map<String, RealMatrix>> setOfPatterns_Testing;
    protected static Map<Integer, String> setOfClasses_Testing;

    protected static Map<Integer, List<Integer>> crossvalMap;

    protected static void trainKNN(String dataset)
            throws IOException, NotEnoughDataException {

        // ==================================================================
        int[] kNeigh = VectorOperations.linearSpace(27, 1, 2);

        RealMatrix errorTrain = new Array2DRowRealMatrix(kNeigh.length, 2);

        for (Integer idx : crossvalMap.keySet()) {

            TrainCrossMultiViewData_MV crossDataKNN = new TrainCrossMultiViewData_MV(
                    setOfPatterns_Training, setOfClasses_Training, crossvalMap, idx);

            // =============== Train and Apply Classifiers
            KNNMultiMatrixMetric knn = new KNNMultiMatrixMetric(
                    crossDataKNN.getSetOfTrainingPatterns(),
                    crossDataKNN.getSetOfTrainingClasses());

            Map<Integer, ClassificationResult> knnWithoutResults = knn.execute(kNeigh,
                    "Missed", crossDataKNN.getSetOfCrossvalPatterns());

            int counter = 0;
            RealMatrix errorMatrix = new Array2DRowRealMatrix(knnWithoutResults.size(), 2);
            for (Integer kValue : knnWithoutResults.keySet()) {

                double errorWithoutNow = ClassifierPerformance
                        .estimateMisclassificationError(knnWithoutResults.get(kValue),
                                crossDataKNN.getSetOfCrossvalClasses());

                errorMatrix.setRow(counter, new double[]{kValue, errorWithoutNow});
                counter++;
            }

            errorTrain = errorTrain.add(errorMatrix);
        }

        errorTrain = errorTrain.scalarMultiply((1 / (double) crossvalMap.keySet().size()));

        for (int idx = 0; idx < kNeigh.length; idx++) {
            LOGGER.info("===========================================");
            LOGGER.info("k-Value: " + errorTrain.getEntry(idx, 0));
            LOGGER.info("Error: " + errorTrain.getEntry(idx, 1));
        }

        List<MLArray> list = new ArrayList<>();
        list.add(new MLDouble("error", errorTrain.getData()));

        MatlabFunctions.storeToFinal("Train-KNN-MultiView-Matrix"+dataset, list);

    }

    protected static void testKNN(int kNeig, String dataset) throws IOException, NotEnoughDataException {

        //===================================================================
        // Try k-NN
        KNNMultiMatrixMetric knn = new KNNMultiMatrixMetric(
                setOfPatterns_Training, setOfClasses_Training);

        // Test
        ClassificationResult knnSelf = knn.execute(kNeig,
                "Missed", setOfPatterns_Training);

        double errorSelf = ClassifierPerformance
                .estimateMisclassificationError(knnSelf,
                        setOfClasses_Training);
        LOGGER.info("Self Test: " + errorSelf);

        ClassificationResult knnWithoutResults = knn.execute(kNeig,
                "Missed", setOfPatterns_Testing);

        double errorWithoutNow = ClassifierPerformance
                .estimateMisclassificationError(knnWithoutResults,
                        setOfClasses_Testing);

        ConfusionMatrix confMatrixWithout = new ConfusionMatrix(setOfClasses_Testing, knnWithoutResults);

        LOGGER.info("===========================================");
        LOGGER.info("With k-NN MultiView, k-NN: " + kNeig);
        LOGGER.info("Without Learned Metric Error: " + errorWithoutNow);
        LOGGER.info("F-Score: " + confMatrixWithout.generateFScore());

        LOGGER.info("===========================================");
        LOGGER.info("With k-NN MultiView, Conf Matrix Without");

        confMatrixWithout.printConfusionMatrix();
        
        confMatrixWithout.printCountMatrix();

        MLArray errorMLWithout = new MLDouble("confMatrix_Without", confMatrixWithout.getConfusionMatrix().getData());
        MLArray countMLWithout = new MLDouble("countMatrix_Without", confMatrixWithout.getCountMatrix().getData());

        List<String> listLabels = confMatrixWithout.getClassTypes();

        MLCell labelsML = new MLCell("labels", new int[]{listLabels.size(), 1});

        int counter = 0;
        for (String label : listLabels) {
            MLChar labelChar = new MLChar("label", label);
            labelsML.set(labelChar, counter);
            counter++;
        }

        List<MLArray> list = new ArrayList<>();
        list.add(errorMLWithout);
        list.add(countMLWithout);
        list.add(labelsML);

        MatlabFunctions.storeToFinal("Test-KNN-MultiView-Matrix" + dataset, list);

    }

}
