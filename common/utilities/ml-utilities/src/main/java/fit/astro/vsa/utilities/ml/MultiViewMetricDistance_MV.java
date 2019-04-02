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
package fit.astro.vsa.utilities.ml;

import fit.astro.vsa.common.bindings.math.ml.metric.MultiViewMetric_MV;
import java.util.Map;
import org.apache.commons.math3.linear.RealMatrix;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class MultiViewMetricDistance_MV {

    private final Map<String, MultiViewMetric_MV> multiMatrix;

    /**
     *
     * @param multiMatrix
     */
    public MultiViewMetricDistance_MV(Map<String, MultiViewMetric_MV> multiMatrix) {
        this.multiMatrix = multiMatrix;
    }

    /**
     *
     * @param x_i
     * @param x_j
     * @return
     */
    public double multiviewDistance(Map<String, RealMatrix> x_i, Map<String, RealMatrix> x_j) {

        double distance = 0;
        for (String variable : multiMatrix.keySet()) {
            
            MetricDistance_MV distance_Matrix = new MetricDistance_MV(
                    multiMatrix.get(variable).getUk(), multiMatrix.get(variable).getVk());
            
            double weight = multiMatrix.get(variable).getWeight();
            distance_Matrix.matrixDistance(x_i.get(variable), x_j.get(variable));

            distance += weight * distance_Matrix.matrixDistance(x_i.get(variable), x_j.get(variable));;
            
        }

        return distance;
    }

}
