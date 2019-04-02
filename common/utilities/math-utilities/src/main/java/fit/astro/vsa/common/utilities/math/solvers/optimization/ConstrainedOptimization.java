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

import fit.astro.vsa.common.utilities.math.NumericalConstants;
import fit.astro.vsa.common.utilities.math.handling.exceptions.IterationTimeOutException;
import org.apache.commons.math3.analysis.UnivariateFunction;

/**
 * Kiefer, J. (1953), "Sequential minimax search for a maximum", Proceedings of
 * the American Mathematical Society, 4 (3): 502–506, doi:10.2307/2032161, JSTOR
 * 2032161, MR 0055639
 * <p>
 * Avriel, Mordecai; Wilde, Douglass J. (1966), "Optimality proof for the
 * symmetric Fibonacci search technique", Fibonacci Quarterly, 4: 265–269, MR
 * 0208812
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class ConstrainedOptimization {

    private static final double TAU = NumericalConstants.GOLDEN_RATIO - 1;
    private static final double EPS_LOCAL = 0.00001;

    // Empty Constructor
    private ConstrainedOptimization() {

    }

    /**
     *
     * @param a left side start
     * @param b right side start
     * @param fin input univariate function
     * <p>
     * @return @throws
     * fit.astro.vsa.common.utilities.math.handling.exceptions.IterationTimeOutException
     */
    public static double GoldenSearchAlgorithm_Minimization(double a, double b,
            UnivariateFunction fin) throws IterationTimeOutException {

        return GoldenSearchAlgorithm_Minimization(a, b, fin, EPS_LOCAL);
    }

    /**
     *
     * @param a left side start
     * @param b right side start
     * @param fin input univariate function
     * <p>
     * @param tol
     *
     * @return
     *
     * @throws
     * fit.astro.vsa.common.utilities.math.handling.exceptions.IterationTimeOutException
     */
    public static double GoldenSearchAlgorithm_Minimization(double a, double b,
            UnivariateFunction fin, double tol) throws IterationTimeOutException {

        // Initialize Golden Search
        double x1 = a + (1 - TAU) * (b - a);
        double f1 = fin.value(x1);

        double x2 = a + TAU * (b - a);
        double f2 = fin.value(x2);

        int counter = 0;

        // Search
        while ((b - a) > tol) {
            if (f1 > f2) {
                a = x1;
                x1 = x2;
                f1 = f2;
                x2 = a + TAU * (b - a);
                f2 = fin.value(x2);
            } else {
                b = x2;
                x2 = x1;
                f2 = f1;
                x1 = a + (1 - TAU) * (b - a);
                f1 = fin.value(x1);
            }
            counter++;

            if (counter > 1000) {
                throw new IterationTimeOutException("Exceeds Iteration.");
            }
        }

        return b;
    }

    /**
     *
     * @param a left side start
     * @param b right side start
     * @param fin input univariate function
     * <p>
     * @return @throws
     * fit.astro.vsa.common.utilities.math.handling.exceptions.IterationTimeOutException
     */
    public static double GoldenSearchAlgorithm_Maximization(double a, double b,
            UnivariateFunction fin) throws IterationTimeOutException {

        return GoldenSearchAlgorithm_Maximization(a, b, fin, EPS_LOCAL);
    }

    /**
     *
     * @param a left side start
     * @param b right side start
     * @param fin input univariate function
     * <p>
     * @param tol
     *
     * @return @throws
     * fit.astro.vsa.common.utilities.math.handling.exceptions.IterationTimeOutException
     */
    public static double GoldenSearchAlgorithm_Maximization(double a, double b,
            UnivariateFunction fin, double tol) throws IterationTimeOutException {

        // Initialize Golden Search
        double x1 = a + (1 - TAU) * (b - a);
        double f1 = fin.value(x1);

        double x2 = a + TAU * (b - a);
        double f2 = fin.value(x2);

        int counter = 0;

        // Search
        while ((b - a) > EPS_LOCAL) {
            if (f1 < f2) {
                a = x1;
                x1 = x2;
                f1 = f2;
                x2 = a + TAU * (b - a);
                f2 = fin.value(x2);
            } else {
                b = x2;
                x2 = x1;
                f2 = f1;
                x1 = a + (1 - TAU) * (b - a);
                f1 = fin.value(x1);
            }
            counter++;

            if (counter > 1000) {
                throw new IterationTimeOutException("Exceeds Iteration.");
            }
        }

        return b;
    }

}
