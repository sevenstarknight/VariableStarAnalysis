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
package fit.astro.vsa.common.datahandling;

import fit.astro.vsa.common.datahandling.LabelHandling;
import fit.astro.vsa.common.utilities.test.classification.GrabIrisData;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.commons.math3.linear.RealVector;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class LabelHandlingTest {

    public LabelHandlingTest() {
    }

    private Map<Integer, RealVector> setOfPatterns;
    private Map<Integer, String> setOfClasses;

    @Before
    public void setUp() throws IOException, URISyntaxException {

        GrabIrisData grabIrisData = new GrabIrisData();
        this.setOfPatterns = grabIrisData.getSetOfPatterns();
        this.setOfClasses = grabIrisData.getSetOfClasses();

    }

    @Test
    public void testLabelHandling() {

        Map<String, Integer> uniqueClasses = LabelHandling.
                countUniqueClasses(setOfClasses);

        List<String> classes = new ArrayList<>(uniqueClasses.keySet());

        assertEquals(50, (int) uniqueClasses.get(classes.get(0)));
        assertEquals(50, (int) uniqueClasses.get(classes.get(1)));
        assertEquals(50, (int) uniqueClasses.get(classes.get(2)));
        
        
        Map<String, List<RealVector>> sortedClasses = LabelHandling.
                sortIntoClasses(setOfPatterns, setOfClasses);

        Map<String, Map<Integer, RealVector>> sortedMaps = LabelHandling.
                sortIntoMaps(setOfPatterns, setOfClasses);

        for (String classTypes : classes) {

            int number = uniqueClasses.get(classTypes);

            List<RealVector> listOfInputs = sortedClasses.get(classTypes);

            Map<Integer, RealVector> mapOfInput = sortedMaps.get(classTypes);

            assertEquals(number, listOfInputs.size());
            assertEquals(number, mapOfInput.size());

        }

    }
}
