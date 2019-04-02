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
package fit.astro.vsa.utilities.ml.cart;

import fit.astro.vsa.common.utilities.io.SerialStorage;
import fit.astro.vsa.common.utilities.test.classification.GrabIrisData;
import fit.astro.vsa.common.bindings.ml.ClassificationResult;
import fit.astro.vsa.utilities.ml.performance.ClassifierPerformance;
import fit.astro.vsa.common.datahandling.training.TestData;
import fit.astro.vsa.common.datahandling.training.TrainCrossData;
import fit.astro.vsa.common.datahandling.training.TrainCrossTestGenerator;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class CARTClassifierAnalysis {

    private static final Random RAND = new Random(42L);
    
    private static final Logger LOGGER
            = LoggerFactory.getLogger(CARTClassifierAnalysis.class);
    
    private static final String resourcePath = "src"
            + File.separator + "test"
            + File.separator + "resources";
    
    private static final String cartLocations = "/datasets/cartExample.ser";
    
    public static void main(String[] args) throws IOException {

        
        GrabIrisData grabIrisData = new GrabIrisData();
        Map<Integer, RealVector> setOfPatterns = grabIrisData.getSetOfPatterns();
        Map<Integer, String> setOfClasses = grabIrisData.getSetOfClasses();

        //=============================================================
        TrainCrossTestGenerator trainTest
                = new TrainCrossTestGenerator(
                        setOfClasses, 0.2, RAND);

        Map<Integer, List<Integer>> crossvalMap = trainTest.getCrossvalMap();

        TrainCrossData crossDataCART = new TrainCrossData(
                setOfPatterns, setOfClasses, crossvalMap, 1);

        // ==================================================================
        // =============== Train and Apply Classifiers
        CARTClassifierGenerator cartGenerator
                = new CARTClassifierGenerator(
                        crossDataCART.getSetOfTrainingPatterns(),
                        crossDataCART.getSetOfTrainingClasses());

        CART cart = cartGenerator.generateCART(0.001,
                crossDataCART.getSetOfCrossvalPatterns(),
                crossDataCART.getSetOfCrossvalClasses());

        TestData td = new TestData(setOfPatterns, setOfClasses,
                trainTest.getTestingData());

        ClassificationResult classificationResult
                = CARTClassifierGenerator.execute(cart,
                        td.getSetOfTestingPatterns());

        double errorCART = ClassifierPerformance.
                estimateMisclassificationError(
                        classificationResult, td.getSetOfTestingClasses());

        
        LOGGER.info("Estimated Misclassification Error of CART Alone: "
                + errorCART);

        LOGGER.info("===========================================");
        LOGGER.info("With Classification and Regression Tree");
        LOGGER.info("Error: " + errorCART);
        
        String path = resourcePath + cartLocations;

        try {
            SerialStorage.storeSerialObject(cart, new File(path));
        } catch (FileNotFoundException exception) {
            LOGGER.error(null, exception);
        }

    }
}
