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
package fit.astro.vsa.utilities.ml.oc;

import fit.astro.vsa.common.utilities.test.classification.GrabIrisData;
import fit.astro.vsa.common.bindings.ml.ClassificationResult;
import fit.astro.vsa.common.datahandling.training.TrainCrossData;
import fit.astro.vsa.common.datahandling.training.TrainCrossTestGenerator;
import fit.astro.vsa.common.datahandling.LabelHandling;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;
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
public class OneClassPWCTest {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(OneClassPWCTest.class);
    
    public OneClassPWCTest() {
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
    public void testOneClassPWC() {

        Map<String, Integer> unique = LabelHandling.countUniqueClasses(setOfClasses);
        String[] uniqueLabels = unique.keySet().toArray(new String[unique.keySet().size()]);

        TrainCrossData crossData = new TrainCrossData(
                setOfPatterns, setOfClasses, crossvalMap, 0);

        Map<Integer, String> reducedTrainingClasses
                = crossData.getSetOfTrainingClasses().entrySet().stream()
                .filter(p -> p.getValue().equals(uniqueLabels[0]) || p.getValue().equals(uniqueLabels[1]))
                .collect(Collectors.toMap(p -> p.getKey(), p -> p.getValue()));

        Map<Integer, RealVector> reducedTrainingData
                = new HashMap<>(crossData.getSetOfTrainingPatterns());
        reducedTrainingData.keySet().retainAll(reducedTrainingClasses.keySet());

        // =============== Train and Apply Classifiers
        OneClassPWC pwc = new OneClassPWC(reducedTrainingData, "Known", "Anomaly");

        double threshold = pwc.train(0.01);
        ClassificationResult pwcWithoutResults = pwc.execute(0.01,
                crossData.getSetOfCrossvalPatterns(), threshold);

        Map<Integer, String> classEstimates
                = pwcWithoutResults.getLabelEstimate();

        // ============= Estimate Error ========================
        double counter = 0;

        for (Integer jdx : crossData.getSetOfCrossvalPatterns().keySet()) {

            if (classEstimates.get(jdx).equalsIgnoreCase("Anomaly")
                    && !crossData.getSetOfCrossvalClasses()
                    .get(jdx).equals(uniqueLabels[2])) {
                counter = counter + 1;
            }

        }

        double error = counter / (double) crossData
                .getSetOfCrossvalPatterns().size();

        assertEquals(0.0, error, 0.01);

    }
}
