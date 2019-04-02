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
package fit.astro.vsa.utilities.ml.ecva;

import fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException;
import fit.astro.vsa.common.utilities.test.classification.GrabIrisData;
import fit.astro.vsa.common.bindings.ml.ClassificationResult;
import fit.astro.vsa.utilities.ml.knn.KNNVectorMetric;
import fit.astro.vsa.utilities.ml.performance.ClassifierPerformance;
import fit.astro.vsa.common.datahandling.training.TrainCrossTestGenerator;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.junit.After;
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
public class ECVATest {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(ECVATest.class);

    private Random RAND = new Random(42L);

    public ECVATest() {
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
    public void testECVA() throws IOException, NotEnoughDataException {

        ECVA ecva = new ECVA(setOfPatterns, setOfClasses);

        CanonicalVariates canonicalVariates = ecva.execute();

        RealMatrix cw = canonicalVariates.getCanonicalWeights();
        RealVector mx = canonicalVariates.getMean();

        Map<Integer, RealVector> canonical = canonicalVariates.getCanonicalVariates();

        KNNVectorMetric knnWith = new KNNVectorMetric(
                canonical,
                setOfClasses);

        ClassificationResult knnWithResults = knnWith.execute(1,
                "Missed", canonical);

        double errorCanon = ClassifierPerformance.
                estimateMisclassificationError(
                        knnWithResults, setOfClasses);

        LOGGER.info("With ECVA: " + errorCanon);
        //===============================================================
        KNNVectorMetric knnWithout = new KNNVectorMetric(
                setOfPatterns, setOfClasses);

        ClassificationResult knnWithoutResults
                = knnWithout.execute(1,
                        "Missed", setOfPatterns);

        double error = ClassifierPerformance.
                estimateMisclassificationError(
                        knnWithoutResults, setOfClasses);

        LOGGER.info("With Out ECVA: " + error);

    }
}
