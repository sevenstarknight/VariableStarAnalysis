/*
 * Copyright (C) 2016 Kyle Johnston
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without isEven the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package fit.astro.vsa.utilities.ml;

import fit.astro.vsa.common.bindings.math.vector.VectorDistanceType;
import fit.astro.vsa.common.utilities.math.NumericTests;
import java.util.ArrayList;
import java.util.List;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
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
public class PairWiseDistanceTest {
    
    public PairWiseDistanceTest() {
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

 
     @Test
     public void testPairWiseDistance() {
     
         List<RealVector> patterns = new ArrayList<>();
         
         patterns.add(MatrixUtils.createRealVector(new double[]{3.0, 3.0}));
         patterns.add(MatrixUtils.createRealVector(new double[]{2.0, 2.0}));
         patterns.add(MatrixUtils.createRealVector(new double[]{1.0, 1.0}));
         
         
         PairWiseDistances distances = new PairWiseDistances(patterns);
         
         RealMatrix distanceMatrix = distances.
                 generateDistances(VectorDistanceType.EUCLIDEAN_DISTANCE);
         
         RealMatrix shouldBe = MatrixUtils.createRealMatrix(3, 3);
         shouldBe.setRow(0, new double[]{0.0, Math.sqrt(2), Math.sqrt(8)});
         shouldBe.setRow(1, new double[]{Math.sqrt(2), 0.0, Math.sqrt(2)});
         shouldBe.setRow(2, new double[]{Math.sqrt(8), Math.sqrt(2), 0.0});
         
         double error = distanceMatrix.subtract(shouldBe).getFrobeniusNorm();
         
         assertEquals(Boolean.TRUE, NumericTests.isApproxZero(error));
     
     }
}
