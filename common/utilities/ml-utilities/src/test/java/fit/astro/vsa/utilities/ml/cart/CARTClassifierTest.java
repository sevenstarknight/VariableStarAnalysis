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
import fit.astro.vsa.common.datahandling.training.TrainCrossData;
import fit.astro.vsa.common.datahandling.training.TrainCrossTestGenerator;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.net.URISyntaxException;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.commons.math3.linear.RealVector;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class CARTClassifierTest {

    private static final Random RAND = new Random(42L);

    private static final Logger LOGGER
            = LoggerFactory.getLogger(CARTClassifierTest.class);

    public CARTClassifierTest() {
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
    public void testCART() throws IOException {

        TrainCrossData crossDataCART = new TrainCrossData(
                setOfPatterns, setOfClasses, crossvalMap, 0);

        // ==================================================================
        // =============== Train and Apply Classifiers
        CARTClassifierGenerator cartGenerator
                = new CARTClassifierGenerator(
                        crossDataCART.getSetOfTrainingPatterns(),
                        crossDataCART.getSetOfTrainingClasses());

        CART cart = cartGenerator.generateCART(0.0001,
                crossDataCART.getSetOfCrossvalPatterns(),
                crossDataCART.getSetOfCrossvalClasses());

        ClassificationResult classificationResult
                = CARTClassifierGenerator.execute(cart,
                        crossDataCART.getSetOfTrainingPatterns());

        double errorCART = ClassifierPerformance.
                estimateMisclassificationError(
                        classificationResult,
                        crossDataCART.getSetOfTrainingClasses());

        LOGGER.info("===========================================");
        LOGGER.info("With CART");
        LOGGER.info("Error: " + errorCART);
        assertEquals(0.1, errorCART, 0.1);
    }

//     @Test
    public void testIO() throws IOException {

        TrainCrossData crossDataCART = new TrainCrossData(
                setOfPatterns, setOfClasses, crossvalMap, 0);

        // ==================================================================
        // =============== Train and Apply Classifiers
        CARTClassifierGenerator cartGenerator
                = new CARTClassifierGenerator(
                        crossDataCART.getSetOfTrainingPatterns(),
                        crossDataCART.getSetOfTrainingClasses());

        CART cart = cartGenerator.generateCART(0.0001,
                crossDataCART.getSetOfCrossvalPatterns(),
                crossDataCART.getSetOfCrossvalClasses());

        // ==================================================================
        // Read Out Works
        boolean isSuccess = Boolean.FALSE;
        String fileName = "/datasets/cartExample.ser";
        try {
            isSuccess = SerialStorage.
                    storeSerialObject(cart, new File(resourcePath + fileName));
        } catch (FileNotFoundException exception) {
            LOGGER.error(null, exception);
        }

        assertEquals(Boolean.TRUE, isSuccess);

        // ==================================================================
        // Read in Works
        boolean cartDoesExist;
        CART cartIN = null;
        ObjectInputStream oisCART;
        try (
                InputStream finCART = CARTClassifierTest.class
                        .getResourceAsStream(fileName)) {
                    oisCART = new ObjectInputStream(finCART);
                    cartIN = (CART) oisCART.readObject();
                    finCART.close();
                    oisCART.close();
                    cartDoesExist = Boolean.TRUE;
                } catch (ClassNotFoundException exception) {
                    LOGGER.error(null, exception);
                    cartDoesExist = Boolean.FALSE;
                }

                assertEquals(Boolean.TRUE, cartIN != null);
                assertEquals(Boolean.TRUE, cartDoesExist);

    }

}
