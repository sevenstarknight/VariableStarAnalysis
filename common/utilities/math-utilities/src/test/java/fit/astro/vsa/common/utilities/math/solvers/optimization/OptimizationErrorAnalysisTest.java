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

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import fit.astro.vsa.common.bindings.math.solver.OptimizationErrorResult;
import org.apache.commons.math3.fitting.WeightedObservedPoint;
import org.apache.commons.math3.linear.MatrixUtils;
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
public class OptimizationErrorAnalysisTest {

    public OptimizationErrorAnalysisTest() {
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

    private final static double[] coeff = new double[]{5.2, 2.3, 1.0};

    @Test
    public void testLevenbergMarquardtErrorAnalysis() {

        Random randomInput = new Random(42L);
        Random randomNoise = new Random(52L);

        List<WeightedObservedPoint> pts = new ArrayList<>();

        for (int idx = 0; idx < 1000; idx++) {
            // Span -2 to 2 uniformly
            double x = randomInput.nextDouble() * 4 - 2;
            double yPlusFuzz = quadraticModel(x, coeff) + (randomNoise.nextDouble() - 0.5);
            pts.add(new WeightedObservedPoint(1.0, x, yPlusFuzz));
        }

        ExampleFitter ef = new ExampleFitter(
                MatrixUtils.createRealVector(new double[]{4.0, 4.0, 4.0}));

        double[] guessCoeff = ef.fit(pts);

        OptimizationErrorAnalysis errorAnalysis
                = new OptimizationErrorAnalysis(new ExampleModel());

        OptimizationErrorResult errorResults
                = errorAnalysis.computeErrors(MatrixUtils.createRealVector(guessCoeff), pts);

        assertEquals(coeff[0], guessCoeff[0],
                errorResults.getCoefficientPredictionError().getEntry(0));

        assertEquals(coeff[1], guessCoeff[1],
                errorResults.getCoefficientPredictionError().getEntry(1));

        assertEquals(coeff[2], guessCoeff[2],
                errorResults.getCoefficientPredictionError().getEntry(2));

    }

    public static double quadraticModel(double x, double[] point) {

        return point[0] * x * x + point[1] * x + point[2];

    }
}
