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
import fit.astro.vsa.common.utilities.math.linearalgebra.VectorOperations;
import org.apache.commons.math3.analysis.function.Abs;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle.Johnston
 */
public class SuperSmootherUnweightedPeriodic extends SuperSmootherMethod {

    private final Real2DCurve series;
    private final double span;
    private final double period;

    /**
     * Smooth non-weighted periodic
     *
     * @param seriesIn
     * @param span
     * @param period
     */
    public SuperSmootherUnweightedPeriodic(Real2DCurve seriesIn, double span,
            double period) {
        this.series = seriesIn;
        this.span = span;
        this.period = period;
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

        RealVector x = new ArrayRealVector(n + k - 1);
        RealVector y = new ArrayRealVector(n + k - 1);

        x.setSubVector(0, series.getXVector());
        x.setSubVector(n, series.getXVector().getSubVector(0, k - 1).mapAdd(
                period));

        y.setSubVector(0, series.getYVector());
        y.setSubVector(n, series.getYVector().getSubVector(0, k - 1));

        // ================================================================
        Real2DCurve seriesSpan = new Real2DCurve(x, y);
        // ================================================================
        RealVector xy = seriesSpan.getYVector().ebeMultiply(seriesSpan.getXVector());

        // ================================================================
        RealMatrix xx = MatrixUtils.createRealMatrix(n + k - 1, 2);
        xx.setColumnVector(0, seriesSpan.getXVector());
        xx.setColumnVector(1, seriesSpan.getXVector());

        // ================================================================
            RealMatrix xxCum = MatrixOperations.cumulativeDimensionalProd(xx, false);

        RealMatrix data = MatrixUtils.createRealMatrix(n + k - 1, 4);
        data.setColumnVector(0, xxCum.getColumnVector(0));
        data.setColumnVector(1, xxCum.getColumnVector(1));
        data.setColumnVector(2, seriesSpan.getYVector());
        data.setColumnVector(3, xy);

        // ================================================================
        // Compute sum(w), sum(w*x), sum(w*x^2), sum(w*y), sum(w*x*y) over k points.
        RealMatrix cumSumData = MatrixOperations.cumulativeDimensionalSummation(data, true);
        RealMatrix cs = MatrixUtils.createRealMatrix(
                cumSumData.getRowDimension() + 1,
                cumSumData.getColumnDimension());
        cs.setRowVector(0, new ArrayRealVector(4)); 
        for (int i = 0; i < cumSumData.getRowDimension(); i++) {
            cs.setRow(i + 1, cumSumData.getRow(i));
        }

        RealMatrix subCs1 = cs.getSubMatrix(k, n + k - 1, 0, 3);
        RealMatrix subCs2 = cs.getSubMatrix(0, n - 1, 0, 3);

        RealMatrix sums = subCs1.subtract(subCs2);

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

        double[] smo = a.add(b.ebeMultiply(seriesSpan.getXVector()
                .getSubVector(m, a.getDimension()))).toArray();

        smo = VectorOperations.circularShift(smo, m);

        // ================================================================
        // acvr
        RealMatrix sums_cv = sums.subtract(data.getSubMatrix(m, n + m - 1, 0, 3));
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

        RealVector smo_cv_tmp = a_cv.add(b_cv.ebeMultiply(seriesSpan.getXVector()
                .getSubVector(m, b_cv.getDimension())));
        RealVector smo_cv = MatrixUtils.createRealVector(VectorOperations.circularShift(smo_cv_tmp.toArray(), m));

        double[] acvr = smo_cv.subtract(
                seriesSpan.getYVector().getSubVector(0, smo_cv.getDimension())).map(new Abs()).toArray();
        // ================================================================
        return new SuperSmootherResults(smo, acvr);
    }

}
