/*
 * Copyright (C) 2016 Kyle Johnston
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without isEven the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package fit.astro.vsa.analysis.feature.smoothing;

import fit.astro.vsa.analysis.feature.smoothing.util.*;
import fit.astro.vsa.common.bindings.math.Real2DCurve;
import fit.astro.vsa.common.utilities.math.NumericTests;
import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * Smoothing of scatterplots using Friedman's supersmoother algorithm.
 *
 * Inputs: X, Y are same-length vectors.
 *
 * Output Y_SMOOTH is a smoothed version of Y.
 *
 * The supersmoother algorithm computes three separate smooth curves from the
 * input data with symmetric spans of 0.05*n, 0.2*n and 0.5*n, where n is the
 * number of data points. The best of the three smooth curves is chosen for each
 * predicted point using leave-one-out cross validation. The best spans are then
 * smoothed by a fixed-span smoother (span = 0.2*n) and the prediction is
 * computed by linearly interpolating between the three smooth curves. This
 * final smooth curve is then smoothed again with a fixed-span smoother (span =
 * 0.05*n).
 *
 * According to comments by Friedman, "For small samples (n lt 40) or if there
 * are substantial serial correlations between observations close in x-value,
 * then a prespecified fixed span smoother (span > 0) should be used. Reasonable
 * span values are 0.2 to 0.4."
 *
 *
 * Variables ----------------------------------------------------------
 *
 * Weights: Vector of relative weights of each data point. Default is for all
 * points to be weighted equally.
 *
 * Span: Sets the width of a fixed-width smoothing operation relative to the
 * number of data points, 0 lt SPAN lt 1. Setting this to be non-zero disables
 * the supersmoother algorithm. Default is 0 (use supersmoother).
 *
 * Period: Sets the period of periodic data. Default is Inf (infinity) which
 * implies that the data is not periodic. Can also be set to zero for the same
 * effect.
 *
 * Alpha: Sets a small-span penalty to produce a greater smoothing effect. 0 lt
 * Alpha lt 10, where 0 does nothing and 10 produces the maximum effect. Default
 * = 0.
 *
 * Friedman, J. H. (1984). A Variable Span Smoother. Tech. Rep. No. 5,
 * Laboratory for Computational Statistics, Dept. of Statistics, Stanford Univ.,
 * California.
 *
 * Version: 1.0, 12 December 2007 Author: Douglas M. Schwarz
 *
 * @author Kyle.Johnston 10-31-2014
 */
public class SuperSmoother {

    private Real2DCurve series;
    private RealVector weights;
    private SuperSmootherProperties prop;

    /**
     * Initialized Super Smoother
     *
     * @param series
     * @param prop
     */
    public SuperSmoother(Real2DCurve series, SuperSmootherProperties prop) {
        if (series.size() < 5) {
            throw new ArithmeticException("Size must be greater than 4");
        }
        

        this.series = series;
        this.weights = series.getWVector();
        this.prop = prop;
    }

    /**
     *
     * @return Smoothed Vector
     */
    public SuperSmootherResults execute() {
        // ===============================================================
        // Compute three smooth curves.
        double[] spans = new double[]{0.05, 0.2, 0.5};
        int nspans = 3;

        // ==================================================================
        int n = series.size();

        // If prop.span > 0 then we have a fixed span smooth.
        if (prop.getSpan() > 0) {
            return smooth(series, prop.getSpan());
        }

        // ===============================================================
        // Compute three smooth curves.
        RealMatrix smo_n = MatrixUtils.createRealMatrix(n, nspans);
        RealMatrix acvr_smo = MatrixUtils.createRealMatrix(n, nspans);

        for (int i = 0; i < nspans; i++) {
            SuperSmootherResults smo1 = smooth(series, spans[i]);
            smo_n.setColumnVector(i, smo1.getSmo_n());

            SuperSmootherResults smo2 = smooth(new Real2DCurve(series.getXVector(), smo1.getResid()),
                    spans[1]);
            acvr_smo.setColumnVector(i, smo2.getSmo_n());
        }

        // ===============================================================
        // Select which smooth curve has smallest error using cross validation.
        double[] minValue = new double[n];
        double[] span_cv = new double[n];
        for (int i = 0; i < n; i++) {
            minValue[i] = acvr_smo.getRowVector(i).getMinValue();
            int minPosition = acvr_smo.getRowVector(i).getMinIndex();
            span_cv[i] = spans[minPosition];
        }

        // ===============================================================
        //  Apply alpha.
        if (prop.getAlpha() != 0.0) {
            double small = 1.0e-7;

            for (int i = 0; i < n; i++) {
                if (minValue[i] < acvr_smo.getEntry(i, 2) && minValue[i] > 0) {
                    span_cv[i] = span_cv[i]
                            + (spans[2] - span_cv[i])
                            * Math.pow(Math.max(small,
                                    minValue[i] / acvr_smo.getEntry(i, 2)),
                                    10 - prop.getAlpha());
                }
            }
        }

        // ===============================================================
        //smooth span_cv and clip at spans(1) and spans(end)
        Real2DCurve tmpSeries
                = new Real2DCurve(series.getXArrayPrimitive(), span_cv);

        SuperSmootherResults smo_span = smooth(tmpSeries, spans[1]);

        for (int i = 0; i < n; i++) {
            smo_span.getSmo_n().setEntry(i, Math.max(
                    Math.min(smo_span.getSmo_n().getEntry(i),
                            spans[spans.length - 1]), spans[0]));
        }

        // ===============================================================
        // Generate CubicSpline
        double[] smo_raw = new double[n];
        for (int i = 0; i < n; i++) {
            Real2DCurve tmpFunction = new Real2DCurve(spans, smo_n.getRow(i));
            
            LinearInterpolator interp = new LinearInterpolator();
            PolynomialSplineFunction interpFunction = interp.interpolate(
                     tmpFunction.getXArrayPrimitive(), tmpFunction.getYArrayPrimitive());
            smo_raw[i] = interpFunction.value(smo_span.getSmo_n().getEntry(i));
             
        }

        // ===============================================================
        tmpSeries = new Real2DCurve(series.getXArrayPrimitive(), smo_raw);
        return smooth(tmpSeries, spans[0]);

    }

    /**
     * Traffic Cop for the Smoothing functions, Each smooth function has been
     * speed optimized for the specific conditions.
     *
     * @param series
     * @param weights
     * @param span
     * @param superSmootherProperties
     * @return
     */
    private SuperSmootherResults smooth(Real2DCurve series, double span) {

        SuperSmootherMethod smootherMethod;

        if (weights != null) {
            if (!NumericTests.isApproxZero(prop.getPeriod())) {
                smootherMethod = new SuperSmootherWeightedPeriodic(
                        series, weights, span, prop.getPeriod());
            } else {
                smootherMethod = new SuperSmootherWeightedAperiodic(
                        series, weights, span);
            }
        } else {
            if (!NumericTests.isApproxZero(prop.getPeriod())) {
                smootherMethod = new SuperSmootherUnweightedPeriodic(
                        series, span, prop.getPeriod());
            } else {
                smootherMethod = new SuperSmootherUnweightedAperiodic(
                        series, span);
            }
        }

        return smootherMethod.execute();
    }

}
