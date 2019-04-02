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

/**
 *
 * @author Kyle Johnston
 */
public class MetricDistance_MV {

    private final RealMatrix uk;
    private final RealMatrix vk;

    
     /**
     * tr{U*(x_i - x_j)'*V*(x_i - x_j)}
     *
     * @param mk
     */
    public MetricDistance_MV(RealMatrix[] mk) {
        this.uk = mk[0];
        this.vk = mk[1];
    }
    
    /**
     * tr{U*(x_i - x_j)'*V*(x_i - x_j)}
     *
     * @param uk
     * @param vk
     */
    public MetricDistance_MV(RealMatrix uk, RealMatrix vk) {
        this.uk = uk;
        this.vk = vk;
    }

    /**
     * 
     * @param x_i
     * @param x_j
     * @return 
     */
    public double matrixDistance(RealMatrix x_i, RealMatrix x_j) {
        return matrixDistance(x_i.subtract(x_j));

    }

    /**
     * 
     * @param deltaij
     * @return 
     */
    public double matrixDistance(RealMatrix deltaij) {

        return (uk.multiply(deltaij.transpose())
                .multiply(vk).multiply(deltaij)).getTrace();

    }

}
