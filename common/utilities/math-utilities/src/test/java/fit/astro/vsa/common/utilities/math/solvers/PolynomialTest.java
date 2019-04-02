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
package fit.astro.vsa.common.utilities.math.solvers;

import fit.astro.vsa.common.utilities.math.solvers.PolynomialSolver;
import fit.astro.vsa.common.utilities.math.handling.exceptions.RootException;
import org.apache.commons.math3.complex.Complex;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Kyle.Johnston
 */
public class PolynomialTest {

    
    private static final Logger LOGGER = LoggerFactory.getLogger(PolynomialSolver.class);

    /**
     *
     */
    @Test
    public void TestPolynomialFunction() {
        LOGGER.info("Testing Polynomial...");

        try {
            // Evaluate p(x) = x^2 + 2*x + 1.
            double[] coef = new double[3];
            coef[0] = 1.;
            coef[1] = 2.;
            coef[2] = 1.;
            PolynomialSolver poly = new PolynomialSolver(coef);

            LOGGER.info("    EXAMPLE 0:  A VERY SIMPLE POLYNOMIAL.");
            LOGGER.info("    p(x) = " + poly);
            LOGGER.info("    p(5) = " + poly.evaluate(5.));
            LOGGER.info("    Zeros of p(x) are:");
            Complex[] zeros = poly.zeros();
            for (int i = 0; i < zeros.length; ++i) {
                LOGGER.info("        " + zeros[i]);
            }

            LOGGER.info("    EXAMPLE 1.  POLYNOMIAL WITH ZEROS 1,2,...,10.");
            coef = new double[11];
            coef[10] = 1;
            coef[9] = -55;
            coef[8] = 1320;
            coef[7] = -18150;
            coef[6] = 157773;
            coef[5] = -902055;
            coef[4] = 3416930;
            coef[3] = -8409500;
            coef[2] = 12753576;
            coef[1] = -10628640;
            coef[0] = 3628800;
            poly = new PolynomialSolver(coef);
            LOGGER.info("    p(x) = " + poly);
            zeros = poly.zeros();
            for (int i = 0; i < zeros.length; ++i) {
                LOGGER.info("        " + zeros[i]);
            }

            LOGGER.info("    EXAMPLE 2. ZEROS ON IMAGINARY AXIS DEGREE 3.");
            Complex[] ccoef = new Complex[4];
            ccoef[3] = new Complex(1, 0);
            ccoef[2] = new Complex(0, -10001.0001);
            ccoef[1] = new Complex(-10001.0001, 0);
            ccoef[0] = new Complex(0, 1);
            poly = new PolynomialSolver(ccoef);
            LOGGER.info("    p(x) = " + poly);
            zeros = poly.zeros();
            for (int i = 0; i < zeros.length; ++i) {
                LOGGER.info("        " + zeros[i]);
            }

            LOGGER.info("    EXAMPLE 3. ZEROS AT 1+I,1/2*(1+I)....1/(2**-9)*(1+I).");
            ccoef = new Complex[11];
            ccoef[10] = new Complex(1., 0.);
            ccoef[9] = new Complex(-1.998046875, -1.998046875);
            ccoef[8] = new Complex(0., 2.658859252929688);
            ccoef[7] = new Complex(0.7567065954208374, -7.567065954208374E-1);
            ccoef[6] = new Complex(-0.2002119533717632, 0.);
            ccoef[5] = new Complex(1.271507365163416E-2, 1.271507365163416E-2);
            ccoef[4] = new Complex(0., -7.820779428584501E-4);
            ccoef[3] = new Complex(-1.154642632172909E-5, 1.154642632172909E-5);
            ccoef[2] = new Complex(1.584803612786345E-7, 0.);
            ccoef[1] = new Complex(-4.652065399568528E-10, -4.652065399568528E-10);
            ccoef[0] = new Complex(0., 9.094947017729282E-13);
            poly = new PolynomialSolver(ccoef);
            LOGGER.info("    p(x) = " + poly);
            zeros = poly.zeros();
            for (int i = 0; i < zeros.length; ++i) {
                LOGGER.info("        " + zeros[i]);
            }

            LOGGER.info("    EXAMPLE 4. MULTIPLE ZEROS.");
            ccoef[10] = new Complex(1, 0);
            ccoef[9] = new Complex(-10, -10);
            ccoef[8] = new Complex(3, 100);
            ccoef[7] = new Complex(284, -334);
            ccoef[6] = new Complex(-1293, 200);
            ccoef[5] = new Complex(2374, 1394);
            ccoef[4] = new Complex(-1587, -3836);
            ccoef[3] = new Complex(-920, 4334);
            ccoef[2] = new Complex(2204, -2352);
            ccoef[1] = new Complex(-1344, 504);
            ccoef[0] = new Complex(288, 0.);
            poly = new PolynomialSolver(ccoef);
            LOGGER.info("    p(x) = " + poly);
            zeros = poly.zeros();
            for (int i = 0; i < zeros.length; ++i) {
                LOGGER.info("        " + zeros[i]);
            }

            LOGGER.info("    EXAMPLE 5. 12 ZEROS EVENLY DISTRIBUTED ON A CIRCLE OF RADIUS"
                    + " 1 CENTERED AT 0+2I.");
            ccoef = new Complex[13];
            ccoef[12] = new Complex(1, 0);
            ccoef[11] = new Complex(0, -24);
            ccoef[10] = new Complex(-264, 0);
            ccoef[9] = new Complex(0, 1760);
            ccoef[8] = new Complex(7920, 0);
            ccoef[7] = new Complex(0, -25344);
            ccoef[6] = new Complex(-59136, 0);
            ccoef[5] = new Complex(0, 101376);
            ccoef[4] = new Complex(126720, 0);
            ccoef[3] = new Complex(0, -112640);
            ccoef[2] = new Complex(-67584, 0);
            ccoef[1] = new Complex(0, 24576);
            ccoef[0] = new Complex(4095, 0);
            poly = new PolynomialSolver(ccoef);
            LOGGER.info("    p(x) = " + poly);
            zeros = poly.zeros();
            for (int i = 0; i < zeros.length; ++i) {
                LOGGER.info("        " + zeros[i]);
            }

        } catch (RootException e) {
            LOGGER.error("context", e);   // Compliant
        }

        // TODO: Assert Equals With Good Values
    }

}
