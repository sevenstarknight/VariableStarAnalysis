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
import fit.astro.vsa.common.utilities.test.classification.GrabMixedViewData;
import fit.astro.vsa.common.bindings.ml.ClassificationResult;
import fit.astro.vsa.common.bindings.ml.metric.MultiViewMetric;
import fit.astro.vsa.utilities.ml.knn.KNNMultiVectorMetric;
import fit.astro.vsa.utilities.ml.metriclearning.l3ml.LargeMarginMultiMetricLearning;
import fit.astro.vsa.utilities.ml.performance.ClassifierPerformance;
import fit.astro.vsa.utilities.ml.training.NormalizeData;
import fit.astro.vsa.common.datahandling.training.TrainCrossTestGenerator;
import fit.astro.vsa.common.datahandling.training.multi.TrainCrossMultiViewData;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.commons.math3.linear.RealVector;
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
public class LargeMarginMultiMetricLearningTest {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(LargeMarginMultiMetricLearningTest.class);

    private final Random RAND = new Random(42L);

    public LargeMarginMultiMetricLearningTest() {
    }

    private Map<Integer, Map<String, RealVector>> setOfPatterns;
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

        GrabMixedViewData grabIrisData = new GrabMixedViewData();
        this.setOfPatterns = grabIrisData.getSetOfPatterns();
        this.setOfClasses = grabIrisData.getSetOfClasses();

        Pair<Map<String, RealVector>, Map<String, RealVector>> transformationVectors
                = NormalizeData.normalizeMultiViewVectorVariate(setOfPatterns);

        setOfPatterns = NormalizeData.applyNormalizeVectorVariate(setOfPatterns,
                transformationVectors);

        //=============================================================
        TrainCrossTestGenerator trainTest
                = new TrainCrossTestGenerator(
                        setOfClasses, 0.25, RAND);

        this.crossvalMap = trainTest.getCrossvalMap();

    }

    @Test
    public void testL3ML() throws IOException, NotEnoughDataException {

        TrainCrossMultiViewData crossDataL3ML = new TrainCrossMultiViewData(
                setOfPatterns, setOfClasses, crossvalMap, 0);

        //===================================================================
        // Try L3ML
        int kNeigh = 7;
        LargeMarginMultiMetricLearning l3ml
                = new LargeMarginMultiMetricLearning(
                        crossDataL3ML.getSetOfTrainingPatterns(),
                        crossDataL3ML.getSetOfTrainingClasses());

        Map<String, MultiViewMetric> outputVar = l3ml.execute(1.0, 5.0);

        // ==================================================================
        double withError = 0;
        double withoutError = 0;

        for (Integer idx : crossvalMap.keySet()) {

            TrainCrossMultiViewData crossDataKNN = new TrainCrossMultiViewData(
                    setOfPatterns, setOfClasses, crossvalMap, idx);

            // =============== Train and Apply Classifiers
            KNNMultiVectorMetric knnWithout = new KNNMultiVectorMetric(
                    crossDataKNN.getSetOfTrainingPatterns(),
                    crossDataKNN.getSetOfTrainingClasses());

            ClassificationResult knnWithoutResults = knnWithout.execute(kNeigh,
                    "Missed", crossDataKNN.getSetOfCrossvalPatterns());

            KNNMultiVectorMetric knnWith = new KNNMultiVectorMetric(
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
        LOGGER.info("With L3ML, kNN -> 1");
        LOGGER.info("With Learned Metric Error: " + withError);
        LOGGER.info("Without Learned Metric Error: " + withoutError);

        assertEquals(Boolean.TRUE, withError <= withoutError);

    }
}
