/*
 * Copyright (C) 2018 Kyle Johnston <kyjohnst2000@my.fit.edu>
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
package fit.astro.vsa.utilities.ml.metriclearning.itml;

import org.apache.commons.math3.linear.RealVector;

/**
 * if (y(i) == y(j)), C(k,:) = [i j 1 l]; else C(k,:) = [i j -1 u]; end
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class ITMLConstraint {

    private final Integer y;
    private final double bound;
    private final RealVector deltaij;

    /**
     *
     * @param y
     * @param bound
     */
    public ITMLConstraint(
            Integer y, double bound, RealVector deltaij) {

        this.y = y;
        this.bound = bound;
        this.deltaij = deltaij;
    }

    public Integer getY() {
        return y;
    }

    public double getBound() {
        return bound;
    }

    public RealVector getDeltaij() {
        return deltaij;
    }
    
    

}
