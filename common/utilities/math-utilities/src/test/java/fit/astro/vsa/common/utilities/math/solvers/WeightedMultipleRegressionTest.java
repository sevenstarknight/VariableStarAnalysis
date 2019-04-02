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

import fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.junit.Test;
import static org.junit.Assert.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Kyle.Johnston
 */
public class WeightedMultipleRegressionTest {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(WeightedMultipleRegressionTest.class);

    /**
     *
     * @throws NotEnoughDataException
     */
    @Test
    public void TestWeightedMultipleRegression() throws NotEnoughDataException {

        RealVector y = MatrixUtils.createRealVector(new double[]{
            4.086543006, 4.345284605, 4.697843221, 4.945264478, 5.218586841,
            5.551340493, 5.889181921, 6.179613335, 6.497139087, 6.759557527,
            7.058689287, 7.378774761, 7.648870268, 7.932156476, 8.282189704,
            8.588736994, 8.804612074, 9.12992595, 9.470309804, 9.780552876
        });

        RealMatrix x = MatrixUtils.createRealMatrix(new double[][]{
            {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,},
            {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,},}).transpose();

        double[] weightArray = new double[]{
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,};

        RealMatrix weight = MatrixUtils.createRealDiagonalMatrix(weightArray);

        LOGGER.info("With 2 coefficients");
        WeightedMultipleRegression lr = new WeightedMultipleRegression(y, x, weight);

        double[] coeff = lr.getCoefficients().toArray();
        LOGGER.info("Coeffecient 0: " + coeff[0]);
        LOGGER.info("Coeffecient 1: " + coeff[1]);
        LOGGER.info("Y = " + coeff[0] + " + " + coeff[1] + " x");

        assertEquals(2, coeff.length);
        assertEquals(4.067938524614291, coeff[0], 0.01);
        assertEquals(2.994021169248118, coeff[1], 0.01);

        LOGGER.info("-------------------------");

        LOGGER.info("i,weight,x,y,estimated_y");
        LOGGER.info("=========================");
        for (int i = 0; i < y.getDimension(); i++) {
            //
            double estimated_y = coeff[0] + coeff[1] * x.getEntry(i, 1);
            LOGGER.info(
                    i + "," + weight.getEntry(i, i) + ","
                    + x.getEntry(i, 1) + "," + y.getEntry(i) + ","
                    + estimated_y
            );
        }

    }

}
