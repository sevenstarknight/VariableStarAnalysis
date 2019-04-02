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
package fit.astro.vsa.utilities.ml.lrc;

import fit.astro.vsa.common.utilities.math.handling.exceptions.IterationTimeOutException;
import fit.astro.vsa.common.utilities.test.classification.GrabIrisData;
import fit.astro.vsa.common.bindings.ml.ClassificationResult;
import fit.astro.vsa.utilities.ml.performance.ClassifierPerformance;
import fit.astro.vsa.common.datahandling.training.TrainCrossData;
import fit.astro.vsa.common.datahandling.training.TrainCrossTestGenerator;
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
public class LRClassifierTest {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(LRClassifierTest.class);
    
    public LRClassifierTest() {
    }


    private Map<Integer, RealVector> setOfPatterns;
    private Map<Integer, String> setOfClasses;

    private Map<Integer, List<Integer>> crossvalMap;

    private TrainCrossTestGenerator trainTest;

    private final Random RAND = new Random(42L);

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
    public void testLRClassifier() throws IOException {

        // ==================================================================
        // =============== Train and Apply Classifiers
        LRClassifierGenerator lrGenerator
                = new LRClassifierGenerator(setOfPatterns, setOfClasses, trainTest);

        TrainCrossData crossData = new TrainCrossData(
                setOfPatterns, setOfClasses, crossvalMap, 0);

        LRC lrc;
        try {
            lrc = lrGenerator.generateLRC();

            // Put Training Data Back into the testing process, should be minimal error
            ClassificationResult classificationResult
                    = LRClassifierGenerator.execute(lrc,
                            crossData.getSetOfTrainingPatterns());

            double error = ClassifierPerformance.
                    estimateMisclassificationError(
                            classificationResult,
                            crossData.getSetOfTrainingClasses());

            assertEquals(0.0, error, 0.1);

            LOGGER.info("Error estimation " + error);
            
        } catch (IterationTimeOutException ex) {
           fail();
        }

    }
}
