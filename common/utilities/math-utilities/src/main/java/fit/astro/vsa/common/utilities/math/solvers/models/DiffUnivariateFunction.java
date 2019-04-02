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

import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunctionGradient;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public interface DiffUnivariateFunction {

    /**
     * Compute the value of the function.
     *
     * @return the ObjectiveFunction.
     */
    ObjectiveFunction getObjectiveFunction();

    /**
     * Compute the gradient of the function with respect to its parameters.
     *
     * @return the ObjectiveFunctionGradient
     */
    ObjectiveFunctionGradient getObjectiveFunctionGradient();

}
