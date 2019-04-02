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
package fit.astro.vsa.analysis.feature;

import fit.astro.vsa.common.bindings.math.Real2DCurve;
import fit.astro.vsa.common.utilities.math.linearalgebra.MatrixOperations;
import fit.astro.vsa.common.utilities.math.linearalgebra.VectorOperations;
import fit.astro.vsa.common.bindings.math.vector.ebe.Atan2ElementFunction;
import org.apache.commons.math3.analysis.function.Exp;
import org.apache.commons.math3.analysis.function.Power;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

/**
 * Algorithm for the application for generating the Lomb-Scargle normalized
 * periodogram
 *
 * https://arxiv.org/abs/1703.09824
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class LombNormalizedPeriodogram {

    //input 
    private final Real2DCurve timeSeriesArray;
    private double overSamplingFactor;
    private double hiSamplingFactor;

    // output
    private Real2DCurve periodogram;

    /**
     * LOMB(T,H,OFAC,HIFAC) computes the Lomb normalized periodogram (spectral
     * power as a function of frequency) of a sequence of N data points H,
     * sampled at times T, which are not necessarily evenly spaced. T and H must
     * be vectors of equal size. The routine will calculate the spectral power
     * for an increasing sequence of frequencies (in reciprocal units of the
     * time array T) up to HIFAC times the average Nyquist frequency, with an
     * oversampling factor of OFAC (typically >= 4).
     * <p>
     * The returned values are arrays of frequencies considered (f), the
     * associated spectral power (P) and estimated significance of the power
     * values (prob). Note: the significance returned is the false alarm
     * probability of the null hypothesis, i.e. that the data is composed of
     * independent Gaussian random variables. Low probability values indicate a
     * high degree of significance in the associated periodic signal.
     * <p>
     * Although this implementation is based on that described in Press,
     * Teukolsky, et al. Numerical Recipes In C, section 13.8, rather than using
     * trigonometric recurrences, this takes advantage of Java's array operators
     * to calculate the exact spectral power as defined in equation 13.8.4 on
     * page 577. This may cause memory issues for large data sets and frequency
     * ranges.
     * <p>
     *
     * Written by Dmitry Savransky 21 May 2008 Translated to Java by Kyle
     * Johnston 14 Jan 2015
     *
     * @param timeSeriesArray
     * @param overSamplingFactor
     * @param hiSamplingFactor
     */
    public LombNormalizedPeriodogram(Real2DCurve timeSeriesArray,
            double overSamplingFactor, double hiSamplingFactor) {

        this.timeSeriesArray = timeSeriesArray;
        this.overSamplingFactor = overSamplingFactor;
        this.hiSamplingFactor = hiSamplingFactor;
    }

    /**
     * Execute the Algorithm and generate the Periodogram
     *
     * @return Real2DCurve , Frequency [Hz] in x vs. Power in y
     */
    public Real2DCurve execute() {

        // Initialize Variables
        int N = timeSeriesArray.size();
        double T = timeSeriesArray.getXVector().getMaxValue()
                - timeSeriesArray.getXVector().getMinValue();

        // totalMean and variance 
        DescriptiveStatistics ds = new DescriptiveStatistics(
                timeSeriesArray.getYArrayPrimitive());

        double mu = ds.getMean();
        double s2 = ds.getVariance();

        //calculate sampling frequencies
        double min = 1.0 / (overSamplingFactor * T);
        double max = hiSamplingFactor * N / (2 * T);

        RealVector fSample = MatrixUtils.createRealVector(VectorOperations.linearSpace(max, min, min));

        //angular frequencies and constant offsets
        RealVector angularFrequency = fSample.mapMultiply(2 * Math.PI);

        RealMatrix wtOuterProduct = angularFrequency.outerProduct(
                timeSeriesArray.getXVector());
        RealMatrix wtOuterProduct2 = wtOuterProduct.scalarMultiply(2.0);

        RealVector x = MatrixOperations.dimensionalSummation(
                MatrixOperations.elementSine(wtOuterProduct2), false);
        RealVector y = MatrixOperations.dimensionalSummation(
                MatrixOperations.elementCosine(wtOuterProduct2), false);

        RealVector tau = VectorOperations.ebeOperations(x, y, new Atan2ElementFunction()).
                mapMultiply(0.5).ebeDivide(angularFrequency);

        RealMatrix repMatrix
                = wtOuterProduct.subtract(MatrixOperations.replicateMatrixColumns(
                        angularFrequency.ebeMultiply(tau), timeSeriesArray.size()));

        // spectral power
        RealMatrix cterm = MatrixOperations.elementCosine(repMatrix);
        RealMatrix sterm = MatrixOperations.elementSine(repMatrix);

        RealMatrix diagHMU = MatrixUtils.createRealDiagonalMatrix(
                timeSeriesArray.getYVector().mapSubtract(mu).toArray());

        RealVector cDHMU2 = MatrixOperations.dimensionalSummation(
                cterm.multiply(diagHMU), false).map(new Power(2));
        RealVector sDHMU2 = MatrixOperations.dimensionalSummation(
                sterm.multiply(diagHMU), false).map(new Power(2));

        RealVector cSumOfSquares = MatrixOperations.dimensionalSummation(
                MatrixOperations.elementSquare(cterm), false);

        RealVector sSumOfSquares = MatrixOperations.dimensionalSummation(
                MatrixOperations.elementSquare(sterm), false);

        RealVector p = cDHMU2.ebeDivide(cSumOfSquares).add(
                sDHMU2.ebeDivide(sSumOfSquares)).mapDivide(2 * s2);

        periodogram = new Real2DCurve(fSample, p);
        return periodogram;

    }

    /**
     *
     * @return RealVector
     */
    public RealVector getSignificanceOfThePower() {

        if (periodogram == null) {
            throw new NullPointerException("Peroidogram not Initialized");
        }

        // estimate of the number of independent frequencies
        double M = 2 * periodogram.size() / overSamplingFactor;
        // statistical significance of power 
        RealVector prob
                = periodogram.getYVector().mapMultiply(-1.0).map(
                        new Exp()).mapMultiply(M);

        for (int i = 0; i < prob.getDimension(); i++) {
            if (prob.getEntry(i) > 0.01) {
                prob.setEntry(i, 1 - Math.pow((1 - Math.exp(-periodogram.getYVector().getEntry(i))), M));
            }
        }

        return prob;
    }

    /**
     * @return the overSamplingFactor
     */
    public double getOverSamplingFactor() {
        return overSamplingFactor;
    }

    /**
     * @param overSamplingFactor the overSamplingFactor to set
     */
    public void setOverSamplingFactor(double overSamplingFactor) {
        this.overSamplingFactor = overSamplingFactor;
    }

    /**
     * @return the hiSamplingFactor
     */
    public double getHiSamplingFactor() {
        return hiSamplingFactor;
    }

    /**
     * @param hiSamplingFactor the hiSamplingFactor to set
     */
    public void setHiSamplingFactor(double hiSamplingFactor) {
        this.hiSamplingFactor = hiSamplingFactor;
    }
}
