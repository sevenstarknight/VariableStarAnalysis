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
package fit.astro.vsa.common.utilities.math.solvers.models;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.fitting.WeightedObservedPoint;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.SimpleValueChecker;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optim.nonlinear.scalar.gradient.NonLinearConjugateGradientOptimizer;
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
public class ProbitDataModelTest {

    public ProbitDataModelTest() {
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

    private final static double[] coeff = new double[]{1.0, 2.0};


    @Test
    public void testProbitModel() {

        Random randomInput = new Random(42L);
        Random randomNoise = new Random(52L);

        List<WeightedObservedPoint> pts = new ArrayList<>();

        for (int idx = 0; idx < 1000; idx++) {
            // Span -2 to 2 uniformly
            double x = randomInput.nextDouble() * 4 - 2;
            double yPlusFuzz = probitModel(x, coeff) + (randomNoise.nextDouble() - 0.5) / 100.0;
            pts.add(new WeightedObservedPoint(1.0, x, yPlusFuzz));
        }

        NonLinearConjugateGradientOptimizer optimizer
                = new NonLinearConjugateGradientOptimizer(
                        NonLinearConjugateGradientOptimizer.Formula.POLAK_RIBIERE,
                        new SimpleValueChecker(1e-10, 1e-10));

        ProbitDataModel probit = new ProbitDataModel(pts);

        PointValuePair optimum = optimizer.optimize(
                new MaxEval(Integer.MAX_VALUE),
                probit.getObjectiveFunction(),
                probit.getObjectiveFunctionGradient(),
                GoalType.MINIMIZE,
                new InitialGuess(new double[]{2, 3}));

        assertEquals(coeff[0], optimum.getPoint()[0], 0.1);
        assertEquals(coeff[1], optimum.getPoint()[1], 0.1);

    }

    private double probitModel(double x, double[] point) {
        NormalDistribution nd
                = new NormalDistribution(point[0], Math.exp(point[1]));

        return nd.cumulativeProbability(x);
    }
}
