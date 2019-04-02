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
import java.util.Collection;
import java.util.List;
import fit.astro.vsa.common.bindings.math.solver.OptimizationErrorResult;
import fit.astro.vsa.common.utilities.math.handling.sigfig.SignificantDigits;
import org.apache.commons.math3.analysis.ParametricUnivariateFunction;
import org.apache.commons.math3.fitting.WeightedObservedPoint;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * Gavin, H.P.(2015) The Levenberg-Marquardt method for nonlinear least squares
 * curve-fitting problems, Dept Civil and Environmental Engineering, Duke
 * University
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class OptimizationErrorAnalysis {

    /**
     * For numData - params > 100 (which should be most of the time)
     */
    private double TVAL95 = 1.96;

    private final ParametricUnivariateFunction inputFunction;

    /**
     * http://people.dukeh.edu/~hpgavin/ce281/lm.pdf
     *
     *
     * @param inputFunction
     */
    public OptimizationErrorAnalysis(ParametricUnivariateFunction inputFunction) {
        this.inputFunction = inputFunction;

    }

    /**
     * Error associated with the coefficients estimated by L-M
     *
     * @param coefficients
     * @param points
     * @return
     */
    public final OptimizationErrorResult computeErrors(
            RealVector coefficients,
            Collection<WeightedObservedPoint> points) {
        RealVector yEstimateVector = MatrixUtils.createRealVector(new double[points.size()]);
        RealVector yObservedVector = MatrixUtils.createRealVector(new double[points.size()]);

        RealMatrix jMatrix = MatrixUtils.createRealMatrix(points.size(),
                coefficients.getDimension());

        RealMatrix wMatrix = MatrixUtils.createRealIdentityMatrix(points.size());

        List<WeightedObservedPoint> listOfPoints = new ArrayList<>(points);

        for (WeightedObservedPoint pt : listOfPoints) {
            int idx = listOfPoints.indexOf(pt);
            //Observed
            yObservedVector.setEntry(idx, pt.getY());

            //Residual
            double yEstimate = inputFunction.value(pt.getX(), coefficients.toArray());
            yEstimateVector.setEntry(idx, yEstimate);

            //Jacobian
            double[] jIdx = inputFunction.gradient(pt.getX(), coefficients.toArray());
            jMatrix.setRow(idx, jIdx);

            //Weights
            wMatrix.setEntry(idx, idx, pt.getWeight());

            idx++;
        }

        //http://people.duke.edu/~hpgavin/ce281/lm.pdf
        RealVector delta = yObservedVector.subtract(yEstimateVector);

        //=====================================================================
        // Equation 20 (Measure of the quality of the fit) Chi-Sq
        double meanSqMeasurementError
                = delta.dotProduct(wMatrix.operate(delta))
                / (points.size() - coefficients.getDimension());

        //=====================================================================
        // Equation 22 (Asymptotic Standard Parameter Errors for the Coefficients
        RealMatrix jTWJ = (jMatrix.transpose().multiply(wMatrix)).multiply(jMatrix);
        RealMatrix jTWJInverse = new LUDecomposition(jTWJ).getSolver().getInverse();

        RealVector coefficientPredictionError
                = MatrixUtils.createRealVector(new double[jTWJInverse.getColumnDimension()]);
        for (int jdx = 0; jdx < jTWJInverse.getColumnDimension(); jdx++) {

            double tmpSqrt = Math.sqrt(jTWJInverse.getEntry(jdx, jdx)) * TVAL95;

            coefficientPredictionError.setEntry(jdx,
                    SignificantDigits.roundSignificantDigitsError(tmpSqrt, 2));
        }

        //==================================================================
        // Equation 24 (Measure of Measurement Error for the Estimate)
        RealMatrix preDiagonal
                = (jMatrix.multiply(jTWJInverse)).multiply(jMatrix.transpose());

        List<WeightedObservedPoint> estimatedFit = new ArrayList<>();
        for (WeightedObservedPoint pt : listOfPoints) {
            int idx = listOfPoints.indexOf(pt);
            double yEstimate = inputFunction.value(pt.getX(),
                    coefficients.toArray());

            double predicitionError
                    = Math.sqrt(preDiagonal.getEntry(idx, idx) + pt.getWeight()) * TVAL95;

            estimatedFit.add(new WeightedObservedPoint(
                    1 / predicitionError, pt.getX(), yEstimate));
        }

        return new OptimizationErrorResult(
                meanSqMeasurementError, coefficientPredictionError,
                estimatedFit);

    }

    public void setTVAL95(double TVAL95) {
        this.TVAL95 = TVAL95;
    }

}
