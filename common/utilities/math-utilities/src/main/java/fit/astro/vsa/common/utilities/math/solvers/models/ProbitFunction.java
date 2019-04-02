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
package fit.astro.vsa.common.utilities.math.solvers.models;

import java.util.Collection;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.fitting.WeightedObservedPoint;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class ProbitFunction implements UnivariateFunction {

    private Collection<WeightedObservedPoint> points;

    private final double mean;
    private final double sigma;

    /**
     * https://en.wikipedia.org/wiki/Probit_model
     *
     * @param coeff
     */
    public ProbitFunction(double[] coeff) {
        this.mean = coeff[0];
        this.sigma = coeff[1];
    }

    /**
     * https://en.wikipedia.org/wiki/Probit_model
     *
     * @param mean - mean value
     * @param sigma - standard deviation
     */
    public ProbitFunction(double mean, double sigma) {
        this.mean = mean;
        this.sigma = sigma;
    }

    ProbitFunction() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    /**
     *
     * @return
     */
    public Collection<WeightedObservedPoint> getPoints() {
        return points;
    }

    /**
     *
     * @param points
     */
    public void setPoints(Collection<WeightedObservedPoint> points) {
        this.points = points;
    }

    /**
     * https://en.wikipedia.org/wiki/Probit_model
     *
     * @param x input magnitude
     * @return detection likelihood
     */
    @Override
    public double value(double x) {

        NormalDistribution normal = new NormalDistribution(mean, Math.exp(sigma));

        return normal.cumulativeProbability(x);
    }

    /**
     *
     * @return
     */
    public double getMean() {
        return mean;
    }

    /**
     *
     * @return
     */
    public double getSigma() {
        return sigma;
    }

}
