/*
 * Copyright (C) 2018 Kyle Johnston 
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without isEven the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package fit.astro.vsa.utilities.ml.metriclearning.ldml;

import fit.astro.vsa.common.utilities.math.NumericTests;
import fit.astro.vsa.utilities.ml.MetricDistance;
import java.util.Map;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston 
 */
public class LDML_MetricLearningGradientGenerator {

    private final Map<Integer, RealVector> ySet;
    private final Map<Integer, RealVector> setOfPatterns;
    // ===============================================
    private double threshold = 0.01;
    private final UnivariateFunction sigmoid = new Sigmoid();

    /**
     * 
     * @param setOfPatterns
     * @param ySet 
     */
    public LDML_MetricLearningGradientGenerator(
            Map<Integer, RealVector> setOfPatterns,
            Map<Integer, RealVector> ySet) {
        this.ySet = ySet;
        this.setOfPatterns = setOfPatterns;

    }

    /**
     *
     * @param lk
     * <p>
     * @return gradiantOfLwrtL
     */
    public RealMatrix execute(RealMatrix lk) {

        RealMatrix metric = lk.transpose().multiply(lk);
        MetricDistance metricDistance = new MetricDistance(metric);

        RealMatrix sumij = MatrixUtils.createRealMatrix(
                lk.getRowDimension(), lk.getColumnDimension());

        for (Integer idx : setOfPatterns.keySet()) {

            RealVector x_i = setOfPatterns.get(idx);

            for (Integer jdx : setOfPatterns.keySet()) {

                RealVector x_j = setOfPatterns.get(jdx);

                RealVector yi = ySet.get(idx);
                RealVector yj = ySet.get(jdx);

                double t_ij = yi.dotProduct(yj);

                double p_ij = sigmoid.value(threshold - 
                        metricDistance.distance(x_i, x_j));

                if(NumericTests.isApproxZero(p_ij)){
                    p_ij = 1e-16;
                }else if(NumericTests.isApproxEqual(p_ij, 1.0)){
                    p_ij = 1.0 - 1e-16;
                }
                
                
                RealVector delta = x_i.subtract(x_j);
                RealMatrix c_ij = delta.outerProduct(delta);

                sumij = sumij.add(c_ij.scalarMultiply(t_ij - p_ij));

            }
        }

        // Equation 4
        return lk.multiply(sumij);
    }

    /**
     * 
     * @param threshold 
     */
    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }

}
