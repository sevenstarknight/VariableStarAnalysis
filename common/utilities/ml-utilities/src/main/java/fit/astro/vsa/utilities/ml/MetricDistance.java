/*
 * Copyright (C) 2018 Kyle Johnston 
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
package fit.astro.vsa.utilities.ml;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston
 */
public class MetricDistance {

    private final RealMatrix metricMatrix;

    /**
     *
     * @param metricMatrix
     */
    public MetricDistance(RealMatrix metricMatrix) {
        this.metricMatrix = metricMatrix;
    }
    
    /**
     * tr{M*(x_i - x_j)(x_i - x_j)'}
     *
     * @param x_i
     * @param x_j
     * @return
     */
    public double traceDistance(RealVector x_i, RealVector x_j) {

        RealVector deltaij = (x_i.subtract(x_j));
        RealMatrix cij = deltaij.outerProduct(deltaij);
        return (metricMatrix.multiply(cij)).getTrace();
    }

    /**
     * (delta_ij)'M(delta_ij)
     *
     * @param deltaij
     * @return
     */
    public double distance(RealVector deltaij) {
        return deltaij.dotProduct(metricMatrix.operate(deltaij));
    }

    /**
     * (x_i - x_j)'M(x_i - x_j)
     *
     * @param x_i
     * @param x_j
     * @return
     */
    public double distance(RealVector x_i, RealVector x_j) {

        RealVector delta = x_i.subtract(x_j);

        return delta.dotProduct(metricMatrix.operate(delta));

    }

    /**
     * sqrt((x_i - x_j)'M(x_i - x_j))
     *
     * @param x_i
     * @param x_j
     * @return
     */
    public double distanceSqrt(RealVector x_i, RealVector x_j) {

        return Math.sqrt(distance(x_i, x_j));

    }
}
