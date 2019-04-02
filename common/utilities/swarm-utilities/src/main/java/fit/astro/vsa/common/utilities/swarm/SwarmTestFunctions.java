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
package fit.astro.vsa.common.utilities.swarm;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author LazoCoder, from https://github.com/LazoCoder
 * @author kjohnston, Updates, clean up, adjustments to Java8
 */
public class SwarmTestFunctions {

    /**
     * Calculate the result of (x^4)-2(x^3). Domain is (-infinity, infinity).
     * Minimum is -1.6875 at x = 1.5.
     *
     * @return function
     */
    public static SwarmOptimizationFunction functionA() {
        return (INDArray input) -> 
                Math.pow(input.getDouble(0), 4) - 2 * Math.pow(input.getDouble(0), 3);
    }

    /**
     * Perform Ackley's function. Domain is [-5, 5] Minimum is 0 at x = 0 & y =
     * 0.
     *
     * @return the function
     */
    public static SwarmOptimizationFunction ackleysFunction() {

        return (INDArray input) -> {
            double x = input.getDouble(0);
            double y = input.getDouble(1);
            double p1 = -20 * Math.exp(-0.2 * Math.sqrt(0.5 * ((x * x) + (y * y))));
            double p2 = Math.exp(0.5 * (Math.cos(2 * Math.PI * x) + Math.cos(2 * Math.PI * y)));
            return p1 - p2 + Math.E + 20;
        };

    }

    /**
     * Perform Booth's function. Domain is [-10, 10] Minimum is 0 at x = 1 & y =
     * 3.
     *
     * @return the function
     */
    public static SwarmOptimizationFunction boothsFunction() {

        return (INDArray input) -> {
            double x1 = input.getDouble(0);
            double y1 = input.getDouble(1);
            double p1 = Math.pow(x1 + 2 * y1 - 7, 2);
            double p2 = Math.pow(2 * x1 + y1 - 5, 2);
            return p1 + p2;
        };

    }

    /**
     * Perform the Three-Hump Camel function.
     *
     * @return the function
     */
    public static SwarmOptimizationFunction threeHumpCamelFunction() {
        return (INDArray input) -> {
            double x1 = input.getDouble(0);
            double y1 = input.getDouble(1);
            double p1 = 2 * x1 * x1;
            double p2 = 1.05 * Math.pow(x1, 4);
            double p3 = Math.pow(x1, 6) / 6;
            return p1 - p2 + p3 + x1 * y1 + y1 * y1;
        };

    }

}
