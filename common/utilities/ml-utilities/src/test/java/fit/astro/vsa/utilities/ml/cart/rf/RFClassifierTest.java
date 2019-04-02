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
package fit.astro.vsa.utilities.ml.cart.rf;

import fit.astro.vsa.common.utilities.test.classification.GrabIrisData;
import fit.astro.vsa.common.bindings.ml.ClassificationResult;
import fit.astro.vsa.common.datahandling.training.TestData;
import fit.astro.vsa.utilities.ml.performance.ClassifierPerformance;
import fit.astro.vsa.common.datahandling.training.TrainCrossTestGenerator;
import fit.astro.vsa.common.datahandling.training.TrainData;
import fit.astro.vsa.utilities.ml.performance.ConfusionMatrix;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.commons.math3.linear.RealVector;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class RFClassifierTest {

    private static final Random RAND = new Random(42L);
    
    private static final Logger LOGGER
            = LoggerFactory.getLogger(RFClassifierTest.class);

    public RFClassifierTest() {
    }

    private final String resourcePath = "src"
            + File.separator + "test"
            + File.separator + "resources";

    private Map<Integer, RealVector> setOfPatterns;
    private Map<Integer, String> setOfClasses;

    private Map<Integer, List<Integer>> crossvalMap;

    private TrainCrossTestGenerator trainTest;


    @Before
    public void setUp() throws IOException, URISyntaxException {


        GrabIrisData grabIrisData = new GrabIrisData();
        this.setOfPatterns = grabIrisData.getSetOfPatterns();
        this.setOfClasses = grabIrisData.getSetOfClasses();

        //=============================================================
        trainTest = new TrainCrossTestGenerator(
                setOfClasses, 0.2, RAND);

        crossvalMap = trainTest.getCrossvalMap();

    }


    @Test
    public void testRF() throws IOException {

        TrainCrossTestGenerator crossDataCART = new TrainCrossTestGenerator(
                setOfClasses, 0.25, RAND);

        TrainData trainingData = new TrainData(setOfPatterns, setOfClasses, 
                crossDataCART.getTrainingData());
        
        TestData testData = new TestData(setOfPatterns, setOfClasses, 
                crossDataCART.getTestingData());
        
        // ==================================================================
        // =============== Train and Apply Classifiers
        RandomForestGenerator rfGenerator
                = new RandomForestGenerator(
                        trainingData.getSetOfTrainingPatterns(), 
                        trainingData.getSetOfTrainingClasses(), 50);
        rfGenerator.setRand(RAND);

        RandomForest rf = rfGenerator.generateRF(
                (int) Math.round(trainingData.getSetOfTrainingClasses().size()*0.75), 0.0001);

        ClassificationResult classificationResult
                = RandomForestGenerator.execute(rf,
                        testData.getSetOfTestingPatterns());

        double errorCART = ClassifierPerformance.
                estimateMisclassificationError(
                        classificationResult,
                        testData.getSetOfTestingClasses());

        LOGGER.info("===========================================");
        LOGGER.info("With RF");
        LOGGER.info("Error: " + errorCART);
        assertEquals(0.1, errorCART, 0.1);

        
        ConfusionMatrix confusionMatrix = new ConfusionMatrix(
                testData.getSetOfTestingClasses(), 
                classificationResult);
        
        LOGGER.info("F-Score: " + confusionMatrix.generateFScore());

    }

}
