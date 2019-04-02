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

import fit.astro.vsa.common.utilities.test.classification.GrabIrisData;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import org.apache.commons.math3.linear.RealVector;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class EMGMMClusteringTest {
    
    
    private static final Logger LOGGER
            = LoggerFactory.getLogger(EMGMMClusteringTest.class);
    
    private Map<Integer, RealVector> setOfPatterns;

    public EMGMMClusteringTest() {
    }

    @BeforeClass
    public static void setUpClass() {
    }

    @AfterClass
    public static void tearDownClass() {
    }

    @Before
    public void setUp() throws IOException {

        GrabIrisData grabIrisData = new GrabIrisData();
        this.setOfPatterns = grabIrisData.getSetOfPatterns();

    }

    @After
    public void tearDown() {
    }

    // TODO add test methods here.
    // The methods must be annotated with annotation @Test. For example:
    //
    @Test
    public void TestGMMEM() {
        
        EMGMMClustering algorithm = new EMGMMClustering(setOfPatterns);
        
        Map<Integer, List<Integer>> resultingAnalysis = algorithm.execute(3);
        
        for(Integer clusterNum : resultingAnalysis.keySet()){
            List<Integer> listOfMembers = resultingAnalysis.get(clusterNum);
            LOGGER.info("Cluster Number: " + clusterNum 
                    + "  contains " + listOfMembers.size() + " elements");
        }
    }
}
