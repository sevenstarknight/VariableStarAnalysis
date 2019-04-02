/*
 * Copyright (C) 2018 Kyle Johnston <kyjohnst2000@my.fit.edu>
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
package fit.astro.vsa.utilities.ml.metriclearning;

import fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException;
import fit.astro.vsa.common.utilities.test.classification.GrabIrisData;
import fit.astro.vsa.common.bindings.ml.ClassificationResult;
import fit.astro.vsa.utilities.ml.knn.KNNVectorMetric;
import fit.astro.vsa.utilities.ml.metriclearning.mmc.MMCLearning;
import fit.astro.vsa.utilities.ml.performance.ClassifierPerformance;
import fit.astro.vsa.common.datahandling.training.TrainCrossData;
import fit.astro.vsa.common.datahandling.training.TrainCrossTestGenerator;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.junit.After;
import org.junit.AfterClass;
import static org.junit.Assert.assertEquals;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class MMCMetricLearningTest {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(MMCMetricLearningTest.class);

    private final Random RAND = new Random(42L);

    public MMCMetricLearningTest() {
    }

    private Map<Integer, RealVector> setOfPatterns;
    private Map<Integer, String> setOfClasses;

    private Map<Integer, List<Integer>> crossvalMap;

    @BeforeClass
    public static void setUpClass() {
    }

    @AfterClass
    public static void tearDownClass() {
    }

    @Before
    public void setUp() throws IOException, URISyntaxException {

        GrabIrisData grabIrisData = new GrabIrisData();
        this.setOfPatterns = grabIrisData.getSetOfPatterns();
        this.setOfClasses = grabIrisData.getSetOfClasses();

        //=============================================================
        TrainCrossTestGenerator trainTest
                = new TrainCrossTestGenerator(
                        setOfClasses, 0.2, RAND);

        crossvalMap = trainTest.getCrossvalMap();

    }

    @After
    public void tearDown() {
    }

    @Test
    public void testMetricLearning() throws IOException, NotEnoughDataException {

        int kNeigh = 7;
        
        TrainCrossData crossData = new TrainCrossData(
                setOfPatterns, setOfClasses, crossvalMap, 0);

        RealMatrix idMatrix = MatrixUtils.createRealIdentityMatrix(
                setOfPatterns.values().iterator().next().getDimension());
        //===================================================================
        // Try MMC
        MMCLearning mmclearning = new MMCLearning(
                        crossData.getSetOfTrainingPatterns(),
                        crossData.getSetOfTrainingClasses());

        RealMatrix mk = mmclearning.generateMetric();

        // ==================================================================
        double withError = 0;
        double withoutError = 0;

        for (Integer idx : crossvalMap.keySet()) {

            TrainCrossData crossDataKNN = new TrainCrossData(
                    setOfPatterns, setOfClasses, crossvalMap, idx);

            // =============== Train and Apply Classifiers
            KNNVectorMetric knnWithout = new KNNVectorMetric(
                    crossDataKNN.getSetOfTrainingPatterns(),
                    crossDataKNN.getSetOfTrainingClasses());

            ClassificationResult knnWithoutResults = knnWithout.execute(kNeigh,
                    "Missed", idMatrix, crossDataKNN.getSetOfCrossvalPatterns());
            
            KNNVectorMetric knnWith = new KNNVectorMetric(
                    crossDataKNN.getSetOfTrainingPatterns(),
                    crossDataKNN.getSetOfTrainingClasses());
            
            ClassificationResult knnWithResults = knnWith.execute(kNeigh,
                    "Missed", mk, crossDataKNN.getSetOfCrossvalPatterns());

            double errorWithNow = ClassifierPerformance
                    .estimateMisclassificationError(knnWithResults,
                            crossDataKNN.getSetOfCrossvalClasses());

            double errorWithoutNow = ClassifierPerformance
                    .estimateMisclassificationError(knnWithoutResults,
                            crossDataKNN.getSetOfCrossvalClasses());

            withError = withError + errorWithNow / (double) crossvalMap.keySet().size();
            withoutError = withoutError + errorWithoutNow / (double) crossvalMap.keySet().size();
        }

        LOGGER.info("===========================================");
        LOGGER.info("With MMC, kNN -> 1");
        LOGGER.info("With Learned Metric Error: " + withError);
        LOGGER.info("Without Learned Metric Error: " + withoutError);

    }

    @Test
    public void testMetricLearning_Diag() throws IOException, NotEnoughDataException {

        TrainCrossData crossData = new TrainCrossData(
                setOfPatterns, setOfClasses, crossvalMap, 0);

        RealMatrix idMatrix = MatrixUtils.createRealIdentityMatrix(
                setOfPatterns.values().iterator().next().getDimension());
        //===================================================================
        // Try MMC
        MMCLearning mmclearning
                = new MMCLearning(
                        crossData.getSetOfTrainingPatterns(),
                        crossData.getSetOfTrainingClasses());

        RealMatrix mk = mmclearning.generateMetric_Diag();

        // ==================================================================
        double withError = 0;
        double withoutError = 0;

        for (Integer idx : crossvalMap.keySet()) {

            TrainCrossData crossDataKNN = new TrainCrossData(
                    setOfPatterns, setOfClasses, crossvalMap, idx);

            // =============== Train and Apply Classifiers
            KNNVectorMetric knn = new KNNVectorMetric(
                    crossDataKNN.getSetOfTrainingPatterns(),
                    crossDataKNN.getSetOfTrainingClasses());

            ClassificationResult knnWithoutResults = knn.execute(1,
                    "Missed", idMatrix, crossDataKNN.getSetOfCrossvalPatterns());
            ClassificationResult knnWithResults = knn.execute(1,
                    "Missed", mk, crossDataKNN.getSetOfCrossvalPatterns());

            Map<Integer, String> hardClassEstimatesWithout
                    = knnWithoutResults.getLabelEstimate();

            Map<Integer, String> hardClassEstimatesWith
                    = knnWithResults.getLabelEstimate();

            // ============= Estimate Error ========================
            double counterWith = 0;
            double counterWithout = 0;

            for (Integer jdx : crossDataKNN.getSetOfCrossvalClasses().keySet()) {
                boolean withMatch
                        = crossDataKNN.getSetOfCrossvalClasses()
                                .get(jdx).equalsIgnoreCase(
                                hardClassEstimatesWith.get(jdx));

                boolean withoutMatch
                        = crossDataKNN.getSetOfCrossvalClasses()
                                .get(jdx).equalsIgnoreCase(
                                hardClassEstimatesWithout.get(jdx));

                if (!withMatch) {
                    counterWith = counterWith + 1;
                }

                if (!withoutMatch) {
                    counterWithout = counterWithout + 1;
                }
            }

            double errorWithNow = counterWith / (double) crossDataKNN.getSetOfCrossvalClasses().size();
            double errorWithoutNow = counterWithout / (double) crossDataKNN.getSetOfCrossvalClasses().size();

            withError = withError + errorWithNow;
            withoutError = withoutError + errorWithoutNow;
        }

        LOGGER.info("===========================================");
        LOGGER.info("With MMC-Diag");
        LOGGER.info("With Learned Metric Error: " + withError / 5.0);
        LOGGER.info("Without Learned Metric Error: " + withoutError / 5.0);

        assertEquals(Boolean.TRUE, withError <= withoutError);
    }

}
