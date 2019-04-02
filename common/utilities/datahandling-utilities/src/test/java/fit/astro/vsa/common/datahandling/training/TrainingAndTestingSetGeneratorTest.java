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
package fit.astro.vsa.common.datahandling.training;

import fit.astro.vsa.common.datahandling.training.TrainCrossTestGenerator;
import fit.astro.vsa.common.utilities.test.classification.GrabIrisData;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.Map;
import java.util.Random;
import org.apache.commons.math3.linear.RealVector;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class TrainingAndTestingSetGeneratorTest {

    public TrainingAndTestingSetGeneratorTest() {
    }
private final Random RAND = new Random(42L);

    private Map<Integer, RealVector> setOfPatterns;
    private Map<Integer, String> setOfClasses;

    @Before
    public void setUp() throws IOException, URISyntaxException {

        GrabIrisData grabIrisData = new GrabIrisData();
        this.setOfPatterns = grabIrisData.getSetOfPatterns();
        this.setOfClasses = grabIrisData.getSetOfClasses();

    }

    @Test
    public void testTrainingAndTestingSetGenerator() {


        TrainCrossTestGenerator generator
                = new TrainCrossTestGenerator(
                         setOfClasses, 0.25, RAND);

        int sizeTraining = generator.getTrainingData().size();
        int sizeTesting = generator.getTestingData().size();

        int total = setOfPatterns.size();

        double ratioTrain = ((double) sizeTraining / (double) total);
        double ratioTest = ((double) sizeTesting / (double) total);

        assertEquals(0.75, ratioTrain, 0.2);
        assertEquals(0.25, ratioTest, 0.2);

        // =======================================
        assertEquals(5, generator.getCrossvalMap().size());

        // =======================================
        assertEquals(0.2, (double) generator.getCrossvalMap().get(0).size() / (double) sizeTraining, 0.1);
        assertEquals(0.2, (double) generator.getCrossvalMap().get(1).size() / (double) sizeTraining, 0.1);
        assertEquals(0.2, (double) generator.getCrossvalMap().get(2).size() / (double) sizeTraining, 0.1);
        assertEquals(0.2, (double) generator.getCrossvalMap().get(3).size() / (double) sizeTraining, 0.1);
        assertEquals(0.2, (double) generator.getCrossvalMap().get(4).size() / (double) sizeTraining, 0.1);
    }
}
