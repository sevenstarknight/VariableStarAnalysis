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
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.fitting.WeightedObservedPoint;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunctionGradient;
import org.apache.commons.math3.random.MersenneTwister;

/**
 * Given a collection of observed points, the class is a representation of the
 * model of the data assuming a probit shape
 * https://en.wikipedia.org/wiki/Probit_model
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class ProbitDataModel implements DiffUnivariateFunction {

    private final Collection<WeightedObservedPoint> data;
    /**
     * https://en.wikipedia.org/wiki/Probit_model
     *
     * @param data observed data
     */
    public ProbitDataModel(Collection<WeightedObservedPoint> data) {
        this.data = data;
    }
    
    public ProbitDataModel(Collection<WeightedObservedPoint> data,
            MersenneTwister rnd) {
        this.data = data;
    }
    

    /**
     *
     * @return the objective function for the Probit
     */
    @Override
    public ObjectiveFunction getObjectiveFunction() {
        return new ObjectiveFunction((double[] params) -> estSSE(params));
    }

    /**
     *
     * @return the objective function for the gradient of the Probit
     */
    @Override
    public ObjectiveFunctionGradient getObjectiveFunctionGradient() {
        return new ObjectiveFunctionGradient((double[] params) -> estGrad(params));
    }

    private double[] estGrad(double[] point) {

        NormalDistribution nd
                = new NormalDistribution(point[0], Math.exp(point[1]));

        double dldmu = 0;
        double dldsig = 0;

        for (WeightedObservedPoint pt : data) {

            double stdNormDensity = nd.density(pt.getX());

            // Delta
            double delta = pt.getY() - nd.cumulativeProbability(pt.getX());
            // Grad (y - y_hat)^2 --> -2(y-y_hat) del(y_hat) /del(mu) 
            dldmu += delta * 2.0 * ((1 / Math.exp(point[1])) * stdNormDensity);

            // Grad (y - y_hat)^2 --> -2(y-y_hat) del(y_hat) /del(sigma) 
            dldsig += delta * 2.0 * (((pt.getX() - point[0]) / Math.exp(point[1])) * stdNormDensity);
        }

        return new double[]{dldmu, dldsig};
    }

    /**
     *
     * @param point
     * @return
     */
    private double estSSE(double[] point) {

        NormalDistribution nd
                = new NormalDistribution(point[0], Math.exp(point[1]));

        double sse = 0;
        for (WeightedObservedPoint pt : data) {
            double delta = pt.getY() - nd.cumulativeProbability(pt.getX());
            sse += pt.getWeight() * delta * delta;
        }

        return sse;
    }

}
