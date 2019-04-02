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
import fit.astro.vsa.common.datahandling.training.TrainCrossData;
import fit.astro.vsa.common.utilities.io.MatlabFunctions;
import fit.astro.vsa.utilities.ml.performance.ClassifierPerformance;
import fit.astro.vsa.utilities.ml.performance.ConfusionMatrix;
import fit.astro.vsa.common.utilities.math.linearalgebra.VectorOperations;
import fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException;
import fit.astro.vsa.utilities.ml.cart.rf.RandomForest;
import fit.astro.vsa.utilities.ml.cart.rf.RandomForestGenerator;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author kjohnston
 */
public class ProcessRawRF {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(ProcessRawRF.class);

    private static final Random RAND = new Random(42L);

    protected static Map<Integer, RealVector> setOfPatterns_Training;
    protected static Map<Integer, String> setOfClasses_Training;

    protected static Map<Integer, RealVector> setOfPatterns_Testing;
    protected static Map<Integer, String> setOfClasses_Testing;

    protected static Map<Integer, List<Integer>> crossvalMap;

    protected static void trainRF(String dataset)
            throws IOException, NotEnoughDataException {

        // ==================================================================
        double[] alphas = VectorOperations.logSpace(0.1, 0.001, 10);

        RealMatrix errorTrain = new Array2DRowRealMatrix(alphas.length, 2);

        for (Integer idx : crossvalMap.keySet()) {

            TrainCrossData crossDataKNN = new TrainCrossData(
                    setOfPatterns_Training, setOfClasses_Training, crossvalMap, idx);

            RandomForestGenerator rfGenerator
                    = new RandomForestGenerator(
                            crossDataKNN.getSetOfTrainingPatterns(),
                            crossDataKNN.getSetOfTrainingClasses(), 50);
            rfGenerator.setRand(RAND);

            int counter = 0;
            RealMatrix errorMatrix = new Array2DRowRealMatrix(alphas.length, 2);
            for (int jdx = 0; jdx < alphas.length; jdx++) {

                RandomForest rf = rfGenerator.generateRF(
                        (int) Math.round(crossDataKNN.getSetOfTrainingPatterns().size() * 0.75), alphas[jdx]);

                ClassificationResult selfTest
                        = RandomForestGenerator.execute(rf,
                                crossDataKNN.getSetOfCrossvalPatterns());

                double errorWithoutNow = ClassifierPerformance
                        .estimateMisclassificationError(selfTest,
                                crossDataKNN.getSetOfCrossvalClasses());

                errorMatrix.setRow(counter, new double[]{alphas[jdx], errorWithoutNow});
                counter++;
            }

            errorTrain = errorTrain.add(errorMatrix);
        }

        errorTrain = errorTrain.scalarMultiply((1 / (double) crossvalMap.keySet().size()));

        for (int idx = 0; idx < alphas.length; idx++) {
            LOGGER.info("===========================================");
            LOGGER.info("alphas-Value: " + errorTrain.getEntry(idx, 0));
            LOGGER.info("Misclassification Error: " + errorTrain.getEntry(idx, 1));
        }

        List<MLArray> list = new ArrayList<>();
        list.add(new MLDouble("error", errorTrain.getData()));

        MatlabFunctions.storeToFinal("Train-RF-MultiView-Vector" + dataset, list);
    }

    protected static void testRF(double alpha, String dataset) throws IOException, NotEnoughDataException {

        //===================================================================
        // Try NCA
        RandomForestGenerator rfGenerator
                = new RandomForestGenerator(
                        setOfPatterns_Training,
                        setOfClasses_Training, 50);
        rfGenerator.setRand(RAND);

        RandomForest rf = rfGenerator.generateRF(
                (int) Math.round(setOfClasses_Training.size() * 0.75), alpha);

        ClassificationResult selfTest
                = RandomForestGenerator.execute(rf,
                        setOfPatterns_Training);

        double errorSelf = ClassifierPerformance
                .estimateMisclassificationError(selfTest,
                        setOfClasses_Training);

        LOGGER.info("Self Test: " + errorSelf);

        ClassificationResult rfResults
                = RandomForestGenerator.execute(rf,
                        setOfPatterns_Testing);

        double errorRF = ClassifierPerformance
                .estimateMisclassificationError(selfTest,
                        setOfClasses_Testing);

        ConfusionMatrix confMatrixWithout = new ConfusionMatrix(setOfClasses_Testing, rfResults);

        LOGGER.info("===========================================");
        LOGGER.info("RF Error: " + errorRF);
        LOGGER.info("F-Score: " + confMatrixWithout.generateFScore());

        LOGGER.info("===========================================");
        LOGGER.info("With RF, Conf Matrix Without");

        confMatrixWithout.printConfusionMatrix();

        LOGGER.info("===========================================");
        LOGGER.info("With  RF, Count Matrix Without");

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

        MatlabFunctions.storeToFinal("Test-RF-SingleView-Vector" + dataset, list);

    }

}
