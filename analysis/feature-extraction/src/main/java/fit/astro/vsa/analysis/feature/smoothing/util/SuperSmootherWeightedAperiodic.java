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
package fit.astro.vsa.analysis.feature.smoothing.util;

import fit.astro.vsa.common.bindings.math.Real2DCurve;
import fit.astro.vsa.common.utilities.math.linearalgebra.MatrixOperations;
import org.apache.commons.math3.analysis.function.Abs;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle.Johnston
 */
public class SuperSmootherWeightedAperiodic extends SuperSmootherMethod {

    private final Real2DCurve series;
    private final RealVector weights;
    private final double span;

    /**
     * Smooth, Weighted, Aperiodic
     *
     * @param series
     * @param weights
     * @param span
     */
    public SuperSmootherWeightedAperiodic(Real2DCurve series,
            RealVector weights,
            double span) {
        this.series = series;
        this.span = span;
        this.weights = weights;
    }

    /**
     *
     * @return
     */
    @Override
    public SuperSmootherResults execute() {

        int n = series.size();
        int m = Math.max((int) Math.round(0.5 * span * n), 2);
        int k = 2 * m + 1;

        // ================================================================
        RealVector wy = weights.ebeMultiply(series.getYVector());
        RealVector wxy = wy.ebeMultiply(series.getXVector());

        // ================================================================
        RealMatrix wxx = MatrixUtils.createRealMatrix(n, 3);
        wxx.setColumnVector(0, weights);
        wxx.setColumnVector(1, series.getXVector());
        wxx.setColumnVector(2, series.getXVector());

        // ================================================================
        RealMatrix wxxCum = MatrixOperations.cumulativeDimensionalProd(wxx, false);

        RealMatrix data = MatrixUtils.createRealMatrix(n, 5);
        data.setColumnVector(0, wxxCum.getColumnVector(0));
        data.setColumnVector(1, wxxCum.getColumnVector(1));
        data.setColumnVector(2, wxxCum.getColumnVector(2));
        data.setColumnVector(3, wy);
        data.setColumnVector(4, wxy);

        // ================================================================
        // Compute sum(w), sum(w*x), sum(w*x^2), sum(w*y), sum(w*x*y) over k points.
        RealMatrix cumSumData = MatrixOperations.cumulativeDimensionalSummation(data, true);
        RealMatrix cs = MatrixUtils.createRealMatrix(n + 1, 5);
        cs.setRowVector(0, new ArrayRealVector(5)); 
        for (int i = 0; i < n; i++) {
            cs.setRow(i + 1, cumSumData.getRow(i));
        }

        RealMatrix subCs1 = cs.getSubMatrix(k, n, 0, 4);
        RealMatrix subCs2 = cs.getSubMatrix(0, n - k, 0, 4);

        RealMatrix sums = MatrixUtils.createRealMatrix(n, 5);
        sums.setSubMatrix(subCs1.subtract(subCs2).getData(), m, 0);

        // ================================================================
        for (int i = 0; i < m; i++) {
            sums.setRowVector(i, sums.getRowVector(m));
        }

        for (int i = n - m; i < n; i++) {
            sums.setRowVector(i, sums.getRowVector(n - m - 1));
        }
        // ================================================================

        RealVector denom = sums.getColumnVector(0).ebeMultiply(
                sums.getColumnVector(2)).subtract(
                        sums.getColumnVector(1).ebeMultiply(
                                sums.getColumnVector(1)));

        RealVector a = sums.getColumnVector(3).ebeMultiply(
                sums.getColumnVector(2)).subtract(
                        sums.getColumnVector(1).ebeMultiply(
                                sums.getColumnVector(4))).ebeDivide(denom);
        RealVector b = sums.getColumnVector(0).ebeMultiply(
                sums.getColumnVector(4)).subtract(
                        sums.getColumnVector(1).ebeMultiply(
                                sums.getColumnVector(3))).ebeDivide(denom);

        double[] smo = a.add(b.ebeMultiply(series.getXVector())).toArray();

        // ================================================================
        // acvr
        RealMatrix sums_cv = sums.subtract(data);
        RealVector denom_cv = sums_cv.getColumnVector(0).ebeMultiply(
                sums_cv.getColumnVector(2)).subtract(
                        sums_cv.getColumnVector(1).ebeMultiply(
                                sums_cv.getColumnVector(1)));

        RealVector a_cv = sums_cv.getColumnVector(3).ebeMultiply(
                sums_cv.getColumnVector(2)).subtract(
                        sums_cv.getColumnVector(1).ebeMultiply(
                                sums_cv.getColumnVector(4))).ebeDivide(denom_cv);
        RealVector b_cv = sums_cv.getColumnVector(0).ebeMultiply(
                sums_cv.getColumnVector(4)).subtract(
                        sums_cv.getColumnVector(1).ebeMultiply(
                                sums_cv.getColumnVector(3))).ebeDivide(denom_cv);

        RealVector smo_cv = a_cv.add(b_cv.ebeMultiply(series.getXVector()));
        double[] acvr = 
                smo_cv.subtract(series.getYVector()).map(new Abs()).toArray();
        // ================================================================
        return new SuperSmootherResults(smo, acvr);
    }

}
