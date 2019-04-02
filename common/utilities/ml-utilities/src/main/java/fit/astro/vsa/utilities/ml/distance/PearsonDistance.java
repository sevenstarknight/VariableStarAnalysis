/*
 * Copyright (C) 2018 kjohnston
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
package fit.astro.vsa.utilities.ml.distance;

import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;

/**
 *
 * @author kjohnston
 */
public class PearsonDistance implements DistanceMeasure {

    private PearsonsCorrelation tmp = new PearsonsCorrelation();

    @Override
    public double compute(double[] arg0, double[] arg1) {
        return 1 - tmp.correlation(arg0, arg1);
    }

}
