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

import org.apache.commons.math3.analysis.ParametricUnivariateFunction;

/*
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class ExampleModel implements ParametricUnivariateFunction {

    /**
     * Example a*x^2 + b*x + c = y
     */
    public ExampleModel() {
    }

    /**
     * a*x^2 + b*x + c = y
     *
     * @param x
     * @param parameters
     * <p>
     * @return
     */
    @Override
    public double value(double x, double... parameters) {

        return parameters[0] * x * x + parameters[1] * x + parameters[2] * 1;
    }

    /**
     * @param parameters
     * @return
     */
    @Override
    public double[] gradient(double x, double... parameters) {

        return new double[]{
            x * x,
            x,
            1
        };
    }

}
