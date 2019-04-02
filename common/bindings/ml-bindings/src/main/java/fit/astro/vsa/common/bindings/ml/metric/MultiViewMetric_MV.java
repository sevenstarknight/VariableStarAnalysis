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
package fit.astro.vsa.common.bindings.math.ml.metric;

import org.apache.commons.math3.linear.RealMatrix;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class MultiViewMetric_MV {

    private final RealMatrix uk;
    private final RealMatrix vk;
    private final double weight;

    public MultiViewMetric_MV(RealMatrix uk, RealMatrix vk, double weight) {
        this.uk = uk;
        this.vk = vk;
        this.weight = weight;
    }

    public RealMatrix getUk() {
        return uk;
    }

    public RealMatrix getVk() {
        return vk;
    }

    public double getWeight() {
        return weight;
    }

}
