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
package fit.astro.vsa.common.bindings.math.vector;

import org.apache.commons.math3.analysis.UnivariateFunction;

/**
 * if greater than zero -> floor; if less than zero -> ceiling (towards zero)
 * 
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class FixFunction implements UnivariateFunction {

    /**
     * Apply fix to elements of the array
     */
    public FixFunction() {

    }

    /**
     * NumericManipulation.fix(x);
     * @param x
     * @return 
     */
    @Override
    public double value(double x) {
        if (x >= 0) {
            return Math.floor(x);
        } else {
            return Math.ceil(x);
        }
    }

}
