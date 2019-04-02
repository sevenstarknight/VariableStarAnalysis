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
package fit.astro.vsa.common.bindings.math.kernel;

import org.apache.commons.math3.analysis.UnivariateFunction;

/**
 * https://en.wikipedia.org/wiki/Kernel_(statistics)
 * 
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class TriangleFunction implements KernelMethod, UnivariateFunction {

    public TriangleFunction() {
    }

    @Override
    public double value(double x) {
        if (x < -1 || x > 1) {
            return 0;
        } else {
            return 1 - Math.abs(x);
        }
    }

}
