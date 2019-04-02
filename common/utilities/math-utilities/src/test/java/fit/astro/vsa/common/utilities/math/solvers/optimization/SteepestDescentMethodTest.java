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
package fit.astro.vsa.common.utilities.math.solvers.optimization;

import org.apache.commons.math3.analysis.UnivariateFunction;
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
public class SteepestDescentMethodTest {

    public SteepestDescentMethodTest() {
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

    private static final double EPS = 1.0e-5;

    @Test
    public void testSteepestDescentMethod() {

        ConvexQuadraticFunction quadFunction
                = new ConvexQuadraticFunction();

        ConvexQuadraticGradFunction quadGradFunction
                = new ConvexQuadraticGradFunction();

        SteepestDescentMethod method = new SteepestDescentMethod(quadFunction, quadGradFunction);

        double result = method.execute(-1.0);

        assertEquals(0.0, result, EPS);
    }

    private static class ConvexQuadraticFunction implements UnivariateFunction {

        public ConvexQuadraticFunction() {
        }

        @Override
        public double value(double x) {
            return x * x;
        }
    }

    private static class ConvexQuadraticGradFunction implements UnivariateFunction {

        public ConvexQuadraticGradFunction() {
        }

        @Override
        public double value(double x) {
            return 2 * x;
        }
    }
}
