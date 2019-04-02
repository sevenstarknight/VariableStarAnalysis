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
package fit.astro.vsa.common.utilities.math.geometry;

import fit.astro.vsa.common.utilities.math.geometry.shapes.PlaneComputations;
import fit.astro.vsa.common.bindings.math.geometry.LineEquation;
import fit.astro.vsa.common.bindings.math.geometry.PlaneEquation;
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
public class PlaneComputationsTest {

    public PlaneComputationsTest() {
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
     * http://www.math.wvu.edu/~hjlai/Teaching/Tip-Pdf/Tip3-10.pdf
     */
    @Test
    public void testPlaneComputations() {

        PlaneEquation planeEquation1 = new PlaneEquation(
                2.0 / 5.0, -1.0 / 5.0, 1.0 / 5.0);

        PlaneEquation planeEquation2 = new PlaneEquation(
                1.0, 1.0, -1.0);

        LineEquation lineEqu = PlaneComputations
                .generatePlanePlaneIntersection(planeEquation1, 
                planeEquation2);
        
        assertEquals(2, lineEqu.getX0().getEntry(0), 1e-6);
        assertEquals(-1, lineEqu.getX0().getEntry(1), 1e-6);
        assertEquals(0, lineEqu.getX0().getEntry(2), 1e-6);
        
        
        
    }
}
