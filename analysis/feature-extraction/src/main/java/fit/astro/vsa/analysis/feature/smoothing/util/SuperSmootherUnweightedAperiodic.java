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
public class SuperSmootherUnweightedAperiodic extends SuperSmootherMethod {

    private final Real2DCurve series;
    private final double span;

    /**
     * Smoothed Non-Weighted Aperiodic
     *
     * @param series
     * @param span
     */
    public SuperSmootherUnweightedAperiodic(Real2DCurve series,
            double span) {
        this.series = series;
        this.span = span;
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

        RealVector xy = series.getXVector().ebeMultiply(series.getYVector());

        // ================================================================
        RealMatrix xx = MatrixUtils.createRealMatrix(n, 2);
        xx.setColumnVector(0, series.getXVector());
        xx.setColumnVector(1, series.getXVector());

        RealMatrix xxCum = MatrixOperations.cumulativeDimensionalProd(xx, false);

        RealMatrix data = MatrixUtils.createRealMatrix(n, 4);
        data.setColumnVector(0, xxCum.getColumnVector(0));
        data.setColumnVector(1, xxCum.getColumnVector(1));
        data.setColumnVector(2, series.getYVector());
        data.setColumnVector(3, xy);

        // ================================================================
        // Compute sum(w), sum(w*x), sum(w*x^2), sum(w*y), sum(w*x*y) over k points.
        RealMatrix cumSumData = MatrixOperations.cumulativeDimensionalSummation(data, true);
        RealMatrix cs = MatrixUtils.createRealMatrix(n + 1, 4);
        cs.setRowVector(0, new ArrayRealVector(4));
        for (int i = 0; i < n; i++) {
            cs.setRow(i + 1, cumSumData.getRow(i));
        }

        RealMatrix subCs1 = cs.getSubMatrix(k, n, 0, 3);
        RealMatrix subCs2 = cs.getSubMatrix(0, n - k, 0, 3);

        RealMatrix sums = MatrixUtils.createRealMatrix(n, 4);
        sums.setSubMatrix(subCs1.subtract(subCs2).getData(), m, 0);

        // ================================================================
        for (int i = 0; i < m; i++) {
            sums.setRowVector(i, sums.getRowVector(m));
        }

        for (int i = n - m; i < n; i++) {
            sums.setRowVector(i, sums.getRowVector(n - m - 1));
        }
        // ================================================================
        RealVector denom = sums.getColumnVector(1).mapMultiply(k).subtract(
                sums.getColumnVector(0).ebeMultiply(
                        sums.getColumnVector(0)));

        RealVector a = sums.getColumnVector(2).ebeMultiply(
                sums.getColumnVector(1)).subtract(
                        sums.getColumnVector(0).ebeMultiply(
                                sums.getColumnVector(3))).ebeDivide(denom);
        RealVector b = sums.getColumnVector(3).mapMultiply(k).subtract(
                sums.getColumnVector(0).ebeMultiply(
                        sums.getColumnVector(2))).ebeDivide(denom);

        double[] smo = a.add(b.ebeMultiply(series.getXVector())).toArray();

        // ================================================================
        // acvr
        RealMatrix sums_cv = sums.subtract(data);
        RealVector denom_cv = sums_cv.getColumnVector(1).mapMultiplyToSelf(k - 1).subtract(
                sums_cv.getColumnVector(0).ebeMultiply(
                        sums_cv.getColumnVector(0)));

        RealVector a_cv = sums_cv.getColumnVector(2).ebeMultiply(
                sums_cv.getColumnVector(1)).subtract(
                        sums_cv.getColumnVector(0).ebeMultiply(
                                sums_cv.getColumnVector(3))).ebeDivide(denom_cv);
        RealVector b_cv = sums_cv.getColumnVector(3).mapMultiplyToSelf(k - 1).subtract(
                sums_cv.getColumnVector(0).ebeMultiply(
                        sums_cv.getColumnVector(2))).ebeDivide(denom_cv);

        RealVector smo_cv = a_cv.add(b_cv.ebeMultiply(series.getXVector()));
        double[] acvr = 
                smo_cv.subtract(series.getYVector()).map(new Abs()).toArray();
        // ================================================================
        return new SuperSmootherResults(smo, acvr);
    }

}
