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

import fit.astro.vsa.common.utilities.math.handling.exceptions.IterationTimeOutException;
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
public class ConstrainedOptimizationFunctionTest {

    public ConstrainedOptimizationFunctionTest() {
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
    public void testGoldenSearchAlgorithm_Minimization() throws IterationTimeOutException {

        ConvexQuadraticFunction quadFunction
                = new ConvexQuadraticFunction();

        double result = ConstrainedOptimization
                .GoldenSearchAlgorithm_Minimization(-2, 2, quadFunction);

        assertEquals(0.0, result, EPS);
    }

    @Test
    public void GoldenSearchAlgorithm_Maximization() throws IterationTimeOutException {

        ConcaveQuadraticFunction quadFunction
                = new ConcaveQuadraticFunction();

        double result = ConstrainedOptimization
                .GoldenSearchAlgorithm_Maximization(-2, 2, quadFunction);

        assertEquals(0.0, result, EPS);
    }

    private static class ConvexQuadraticFunction implements UnivariateFunction {

        public ConvexQuadraticFunction() {
        }

        @Override
        public double value(double x) {
            return x * x - 2;
        }
    }

    private static class ConcaveQuadraticFunction implements UnivariateFunction {

        public ConcaveQuadraticFunction() {
        }

        @Override
        public double value(double x) {
            return -x * x + 2;
        }
    }
}
