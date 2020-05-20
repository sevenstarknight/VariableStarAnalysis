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
import fit.astro.vsa.common.bindings.ml.ClassificationResult;
import fit.astro.vsa.common.bindings.ml.metric.MultiViewMetric_MV;
import fit.astro.vsa.utilities.ml.performance.ClassifierPerformance;
import fit.astro.vsa.utilities.ml.training.NormalizeData;
import fit.astro.vsa.common.datahandling.training.TrainCrossTestGenerator;
import fit.astro.vsa.common.datahandling.training.multi.TrainCrossMultiViewData_MV;
import fit.astro.vsa.common.utilities.test.classification.GrabMixedViewMatrixData;
import fit.astro.vsa.utilities.ml.knn.KNNMultiMatrixMetric;
import fit.astro.vsa.utilities.ml.metriclearning.l3ml_mv.LargeMarginMultiMetricLearning_MV;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.Pair;
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
public class LargeMarginMultiMetricLearning_MVTest {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(LargeMarginMultiMetricLearning_MVTest.class);

    private final Random RAND = new Random(42L);

    public LargeMarginMultiMetricLearning_MVTest() {
    }

    private Map<Integer, Map<String, RealMatrix>> setOfPatterns_Training;
    private Map<Integer, String> setOfClasses_Training;

    private Map<Integer, Map<String, RealMatrix>> setOfPatterns_Testing;
    private Map<Integer, String> setOfClasses_Testing;

    private Map<Integer, List<Integer>> crossvalMap;

    @BeforeClass
    public static void setUpClass() {
    }

    @AfterClass
    public static void tearDownClass() {
    }

    @Before
    public void setUp() throws IOException, URISyntaxException {

        GrabMixedViewMatrixData grabIrisData = new GrabMixedViewMatrixData();
        Map<Integer, Map<String, RealMatrix>> setOfPatterns = grabIrisData.getSetOfPatterns();
        Map<Integer, String> setOfClasses = grabIrisData.getSetOfClasses();

        //=============================================================
        TrainCrossTestGenerator trainTest
                = new TrainCrossTestGenerator(
                        setOfClasses, 0.5, RAND);

        this.crossvalMap = trainTest.getCrossvalMap();

        //=============================================================
        List<Integer> keysTraining = trainTest.getTrainingData();

        Map<Integer, Map<String, RealMatrix>> setOfPatterns_Training_Tmp
                = keysTraining.stream().filter(setOfPatterns::containsKey)
                        .collect(Collectors.toMap(Function.identity(), setOfPatterns::get));

        Pair<Map<String, RealMatrix>, Map<String, RealMatrix>> transformationVectors
                = NormalizeData.normalizeMultiViewMatrixVariate(setOfPatterns_Training_Tmp);

        setOfPatterns_Training = NormalizeData.applyNormalizeMatrixVariate(setOfPatterns_Training_Tmp,
                transformationVectors);

        setOfClasses_Training = keysTraining.stream().filter(setOfClasses::containsKey)
                .collect(Collectors.toMap(Function.identity(), setOfClasses::get));

        //=============================================================
        List<Integer> keysTesting = trainTest.getTestingData();

        Map<Integer, Map<String, RealMatrix>> setOfPatterns_Testing_Tmp
                = keysTesting.stream().filter(setOfPatterns::containsKey)
                        .collect(Collectors.toMap(Function.identity(), setOfPatterns::get));

        setOfPatterns_Testing = NormalizeData.applyNormalizeMatrixVariate(setOfPatterns_Testing_Tmp,
                transformationVectors);

        setOfClasses_Testing = keysTesting.stream().filter(setOfClasses::containsKey)
                .collect(Collectors.toMap(Function.identity(), setOfClasses::get));

    }

    @Test
    public void testL3ML() throws IOException, NotEnoughDataException {

        TrainCrossMultiViewData_MV tmp = new TrainCrossMultiViewData_MV(
                setOfPatterns_Training, setOfClasses_Training, crossvalMap, 0);

        // Try L3ML
        int kNeigh = 7;
        LargeMarginMultiMetricLearning_MV l3ml
                = new LargeMarginMultiMetricLearning_MV(
                        tmp.getSetOfTrainingPatterns(),
                        tmp.getSetOfTrainingClasses());

        Map<String, MultiViewMetric_MV> outputVar = l3ml.execute(15);

        // ==================================================================
        double withError = 0;
        double withoutError = 0;

        for (Integer idx : crossvalMap.keySet()) {

            TrainCrossMultiViewData_MV crossDataKNN = new TrainCrossMultiViewData_MV(
                    setOfPatterns_Training, setOfClasses_Training, crossvalMap, idx);

            // =============== Train and Apply Classifiers
            KNNMultiMatrixMetric knnWithout = new KNNMultiMatrixMetric(
                    crossDataKNN.getSetOfTrainingPatterns(),
                    crossDataKNN.getSetOfTrainingClasses());

            ClassificationResult knnWithoutResults = knnWithout.execute(kNeigh,
                    "Missed", crossDataKNN.getSetOfCrossvalPatterns());

            KNNMultiMatrixMetric knnWith = new KNNMultiMatrixMetric(
                    crossDataKNN.getSetOfTrainingPatterns(),
                    crossDataKNN.getSetOfTrainingClasses());

            ClassificationResult knnWithResults = knnWith.execute(kNeigh,
                    "Missed", outputVar, crossDataKNN.getSetOfCrossvalPatterns());

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
        LOGGER.info("With L3ML, kNN -> 7");
        LOGGER.info("With Learned Metric Error: " + withError);
        LOGGER.info("Without Learned Metric Error: " + withoutError);

        assertEquals(Boolean.TRUE, withError <= withoutError);

    }

}
