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
package fit.astro.vsa.analysis.df;

import fit.astro.vsa.analysis.df.generators.DFGenerator;
import fit.astro.vsa.analysis.df.generators.DFOptions;
import fit.astro.vsa.common.bindings.ml.TimeDomainAttributeMaps;
import fit.astro.vsa.common.bindings.math.Real2DCurve;
import fit.astro.vsa.utilities.ml.ecva.CanonicalVariates;
import fit.astro.vsa.utilities.ml.ecva.ECVA;
import fit.astro.vsa.common.utilities.io.ReadingInUCRData;
import fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException;
import fit.astro.vsa.common.utilities.math.linearalgebra.MatrixOperations;
import fit.astro.vsa.common.bindings.ml.ClassificationResult;
import fit.astro.vsa.utilities.ml.knn.KNNVectorMetric;
import fit.astro.vsa.utilities.ml.metriclearning.lmnn.LargeMarginNearestNeighbor;
import fit.astro.vsa.utilities.ml.metriclearning.nca.NeighbourhoodComponentsAnalysis;
import fit.astro.vsa.common.datahandling.training.TrainCrossData;
import fit.astro.vsa.common.datahandling.training.TrainCrossTestGenerator;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author SevenStarKnight
 */
public class DFMetricLearningTest {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(DFMetricLearningTest.class);

    private Random RAND = new Random(42L);

    public DFMetricLearningTest() {
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

        ReadingInUCRData starLight = new ReadingInUCRData(
                "/Users/kjohnston/Google Drive/VarStarData/UCR");

        TimeDomainAttributeMaps trainingData = starLight.getTrainData("StarLightCurves");
        TimeDomainAttributeMaps testingData = starLight.getTestData("StarLightCurves");

        // =======================================================

        DFOptions dFOptions = new DFOptions(25, 35,
                new int[]{7, 1}, 0.4, DFOptions.Directions.both);

        DFGenerator dfGenerator
                = new DFGenerator(dFOptions);

        Map<Integer, RealVector> tmpPattern = new HashMap<>();

        this.setOfPatterns = new HashMap<>();
        this.setOfClasses = new HashMap<>();

        for (Integer idx : trainingData.getSetOfWaveforms().keySet()) {

            Real2DCurve currentWaveform = trainingData.getSetOfWaveforms().get(idx);

            RealMatrix df = dfGenerator.evaluate(currentWaveform);

            DescriptiveStatistics descript = new DescriptiveStatistics(currentWaveform
                    .getYArrayPrimitive());

            tmpPattern.put(idx, (new ArrayRealVector(
                    MatrixOperations.unpackMatrix(df)))
                    .append(descript.getMean())
                    .append(descript.getStandardDeviation()));

            setOfClasses.put(idx, trainingData.getSetOfClasses().get(idx));
        }

        ECVA ecva = new ECVA(tmpPattern, setOfClasses);

        CanonicalVariates canonicalVariates = ecva.execute();

        setOfPatterns = canonicalVariates.getCanonicalVariates();
    }

    @Test
    public void testNCA() throws IOException, NotEnoughDataException {

        //=============================================================
        TrainCrossTestGenerator trainTest
                = new TrainCrossTestGenerator(
                        setOfClasses, 0.2, RAND);

        this.crossvalMap = trainTest.getCrossvalMap();

        TrainCrossData crossDataNCA = new TrainCrossData(
                setOfPatterns, setOfClasses, crossvalMap, 0);

        //===================================================================
        // Try NCA
        NeighbourhoodComponentsAnalysis nca
                = new NeighbourhoodComponentsAnalysis(
                        crossDataNCA.getSetOfTrainingPatterns(),
                        crossDataNCA.getSetOfTrainingClasses());

        RealMatrix mk = nca.generateMetric();

        // ==================================================================
        double withError = 0;
        double withoutError = 0;

        ClassificationResult knnWithoutResults = null;
        ClassificationResult knnWithResults = null;

        for (Integer idx : crossvalMap.keySet()) {

            TrainCrossData crossDataKNN = new TrainCrossData(
                    setOfPatterns, setOfClasses, crossvalMap, idx);

            // =============== Train and Apply Classifiers
            KNNVectorMetric knn = new KNNVectorMetric(
                    crossDataKNN.getSetOfTrainingPatterns(),
                    crossDataKNN.getSetOfTrainingClasses());

            knnWithoutResults = knn.execute(1,
                    "Missed", crossDataKNN.getSetOfCrossvalPatterns());
            knnWithResults = knn.execute(1,
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

            double errorWithNow = counterWith
                    / (double) crossDataKNN.getSetOfCrossvalClasses().size();
            double errorWithoutNow = counterWithout
                    / (double) crossDataKNN.getSetOfCrossvalClasses().size();

            withError = withError + errorWithNow;
            withoutError = withoutError + errorWithoutNow;
        }

        LOGGER.info("===========================================");
        LOGGER.info("With NCA");
        LOGGER.info("With Learned Metric Error: " + withError / 5.0);
        LOGGER.info("Without Learned Metric Error: " + withoutError / 5.0);

        LOGGER.info("===========================================");

    }

//    @Test
    public void testLMNN() throws IOException, NotEnoughDataException {

        //=============================================================
        TrainCrossTestGenerator trainTest
                = new TrainCrossTestGenerator(
                        setOfClasses, 0.2, RAND);

        this.crossvalMap = trainTest.getCrossvalMap();

        TrainCrossData crossDataLMNN = new TrainCrossData(
                setOfPatterns, setOfClasses, crossvalMap, 0);

        //===================================================================
        // Try LMNN
        int kNeigh = 7;
        LargeMarginNearestNeighbor lmnn
                = new LargeMarginNearestNeighbor(
                        crossDataLMNN.getSetOfTrainingPatterns(),
                        crossDataLMNN.getSetOfTrainingClasses(), kNeigh);

        RealMatrix mk = lmnn.generateMetric();

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

            ClassificationResult knnWithoutResults = knn.execute(kNeigh,
                    "Missed", crossDataKNN.getSetOfCrossvalPatterns());
            ClassificationResult knnWithResults = knn.execute(kNeigh,
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

            double errorWithNow = counterWith
                    / (double) crossDataKNN.getSetOfCrossvalClasses().size();
            double errorWithoutNow = counterWithout
                    / (double) crossDataKNN.getSetOfCrossvalClasses().size();

            withError = withError + errorWithNow;
            withoutError = withoutError + errorWithoutNow;
        }

        LOGGER.info("===========================================");
        LOGGER.info("With LMNN");
        LOGGER.info("With Learned Metric Error: " + withError / 5.0);
        LOGGER.info("Without Learned Metric Error: " + withoutError / 5.0);

    }
}
