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

import java.util.logging.Level;
import java.util.logging.Logger;
import fit.astro.vsa.common.utilities.math.NumericalConstants;
import fit.astro.vsa.common.utilities.math.handling.exceptions.IterationTimeOutException;
import org.apache.commons.math3.analysis.UnivariateFunction;

/**
 * https://en.wikipedia.org/wiki/Gradient_descent
 * 
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class SteepestDescentMethod {

    private final UnivariateFunction optimizationFunction;
    private final UnivariateFunction gradOfOptimization;

    /**
     * 
     * @param optimizationFunction objective function
     * @param gradOfOptimization grad of the objective
     */
    public SteepestDescentMethod(UnivariateFunction optimizationFunction, 
            UnivariateFunction gradOfOptimization) {

        this.optimizationFunction = optimizationFunction;
        this.gradOfOptimization = gradOfOptimization;
    }

    /**
     * 
     * @param guess
     * @return 
     */
    public double execute(double guess) {

        double alphak = 10;
        while (optimizationFunction.value(guess) > NumericalConstants.EPS) {

            AdjustGuess adjustGuess = new AdjustGuess(optimizationFunction, gradOfOptimization, guess);

            try {
                alphak = ConstrainedOptimization.GoldenSearchAlgorithm_Minimization(
                        0.0001, 1, adjustGuess);
            } catch (IterationTimeOutException ex) {
                Logger.getLogger(SteepestDescentMethod.class.getName()).log(Level.SEVERE, null, ex);
            }

            double delta = gradOfOptimization.value(guess) * alphak;

            guess = guess - delta;
        }

        return guess;
    }

    /**
     * 
     */
    private static class AdjustGuess implements UnivariateFunction {

        private final UnivariateFunction fin;
        private final UnivariateFunction finGrad;
        private final double guess;

        public AdjustGuess(UnivariateFunction fin, UnivariateFunction finGrad, double guess) {
            this.fin = fin;
            this.finGrad = finGrad;
            this.guess = guess;
        }

        @Override
        public double value(double x) {
            double newGuess = guess - finGrad.value(guess) * x;

            return fin.value(newGuess);

        }

    }

}
