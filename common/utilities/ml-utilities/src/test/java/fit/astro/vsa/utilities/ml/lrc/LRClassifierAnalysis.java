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
import fit.astro.vsa.common.datahandling.training.TestData;
import fit.astro.vsa.common.datahandling.training.TrainCrossTestGenerator;
import java.io.IOException;
import java.util.Map;
import java.util.Random;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Kyle Johnston
 */
public class LRClassifierAnalysis {

    private static final Random RAND = new Random(42L);
    
    private static final Logger LOGGER
            = LoggerFactory.getLogger(LRClassifierAnalysis.class);
    
    public static void main(String[] args) throws IOException {
        
        GrabIrisData grabIrisData = new GrabIrisData();
        Map<Integer, RealVector> setOfPatterns = grabIrisData.getSetOfPatterns();
        Map<Integer, String> setOfClasses = grabIrisData.getSetOfClasses();

        TrainCrossTestGenerator trainTest
                = new TrainCrossTestGenerator(
                        setOfClasses, 0.20, RAND);

        // ==================================================================
        // =============== Train and Apply Classifiers
        LRClassifierGenerator lrGenerator
                = new LRClassifierGenerator(setOfPatterns, setOfClasses, trainTest);

        TestData td = new TestData(setOfPatterns, setOfClasses,
                trainTest.getTestingData());
        
        LRC lrc;
        try {
            lrc = lrGenerator.generateLRC();
            
            ClassificationResult classificationResult
                    = LRClassifierGenerator.execute(lrc,
                            td.getSetOfTestingPatterns());

            double error = ClassifierPerformance.
                    estimateMisclassificationError(
                            classificationResult, 
                            td.getSetOfTestingClasses());

            // ==================================================================
            
            LOGGER.info("===========================================");
            LOGGER.info("With Logistic Regression Classifier");
            LOGGER.info("Error: " + error);

        } catch (IterationTimeOutException ex) {
            LOGGER.warn(ex.getMessage());
        }

    }

}
