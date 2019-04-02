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
package fit.astro.vsa.common.utilities.clustering;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Random;
import fit.astro.vsa.common.utilities.test.classification.GrabIrisData;
import org.apache.commons.math3.linear.RealVector;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class KMeansClusteringTest {

    public KMeansClusteringTest() {
    }

    @BeforeClass
    public static void setUpClass() {
    }

    @AfterClass
    public static void tearDownClass() {
    }

    @Before
    public void setUp() {
    }

    @After
    public void tearDown() {
    }

    /**
     * 
     * @throws IOException 
     */
    @Test
    public void testKMeans() throws IOException {

        Random rnd = new Random(42L);
        
        GrabIrisData grabIrisData = new GrabIrisData();
        Map<Integer, RealVector> setOfPatterns = grabIrisData.getSetOfPatterns();
       
        KMeansClustering algorithm = new KMeansClustering(setOfPatterns, rnd);

        Map<Integer, List<Integer>> resultingAnalysis = algorithm.execute(3);

        assertEquals(3, resultingAnalysis.keySet().size());

        assertEquals(50, resultingAnalysis.get(0).size());
        assertEquals(39, resultingAnalysis.get(1).size());
        assertEquals(61, resultingAnalysis.get(2).size());

    }
}
