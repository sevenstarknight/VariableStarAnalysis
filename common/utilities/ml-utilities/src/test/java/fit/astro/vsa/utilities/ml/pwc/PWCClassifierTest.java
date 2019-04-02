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
package fit.astro.vsa.utilities.ml.pwc;

import fit.astro.vsa.common.bindings.math.kernel.KernelType;
import fit.astro.vsa.common.utilities.test.classification.GrabIrisData;
import fit.astro.vsa.common.bindings.ml.ClassificationResult;
import fit.astro.vsa.common.datahandling.training.TrainCrossData;
import fit.astro.vsa.common.datahandling.training.TrainCrossTestGenerator;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.commons.math3.linear.RealVector;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author SevenStarKnight
 */
public class PWCClassifierTest {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(PWCClassifierTest.class);
    
    private final Random RAND = new Random(42L);
    
    public PWCClassifierTest() {
    }

    private Map<Integer, RealVector> setOfPatterns;
    private Map<Integer, String> setOfClasses;

    private Map<Integer, List<Integer>> crossvalMap;


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
    public void testPWC() throws IOException {

        // ==================================================================
        double error = 0;

        for (Integer idx : crossvalMap.keySet()) {

            TrainCrossData crossDataPWC = new TrainCrossData(
                    setOfPatterns, setOfClasses, crossvalMap, idx);

            // =============== Train and Apply Classifiers
            PWCClassifier pwc = new PWCClassifier(
                    crossDataPWC.getSetOfTrainingPatterns(),
                    crossDataPWC.getSetOfTrainingClasses());

            ClassificationResult knnWithoutResults = pwc.execute(KernelType.GAUSSIAN,
                    0.5, "Missed", crossDataPWC.getSetOfCrossvalPatterns());

            Map<Integer, String> classEstimates
                    = knnWithoutResults.getLabelEstimate();

            // ============= Estimate Error ========================
            double counter = 0;

            for (Integer jdx : crossDataPWC.getSetOfCrossvalPatterns().keySet()) {
                boolean isMatch
                        = crossDataPWC.getSetOfCrossvalClasses()
                        .get(jdx).equalsIgnoreCase(
                                classEstimates.get(jdx));

                if (!isMatch) {
                    counter = counter + 1;
                }
            }

            double errorPart = counter
                    / (double) crossDataPWC.getSetOfCrossvalPatterns().size();

            error = error + errorPart;
        }

        LOGGER.info("===========================================");
        LOGGER.info("With PWC");
        LOGGER.info("Error: " + error / 5.0);

    }
}
