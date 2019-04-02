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
package fit.astro.vsa.utilities.ml.metriclearning.lmnn;

import fit.astro.vsa.utilities.ml.MetricDistance;
import fit.astro.vsa.utilities.ml.utils.SupportingFunctionality;
import java.util.List;
import java.util.Map;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * Song, Kun, et al. "Parameter Free Large Margin Nearest Neighbor for Distance
 * Metric Learning." AAAI. 2017.
 *
 * @author Kyle Johnston
 */
public class LMNN_MetricLearningGradientGenerator {

    private final Map<Integer, RealVector> mapOfPatterns;
    private final Map<Integer, String> mapOfClasses;

    // =======================================
    private final Map<Integer, List<Integer>> classMemberNear;

    private double GAMMA = 0.5;

    /**
     *
     * @param classMemberNear
     * @param mapOfPatterns
     * @param mapOfClasses
     */
    public LMNN_MetricLearningGradientGenerator(
            Map<Integer, List<Integer>> classMemberNear,
            Map<Integer, RealVector> mapOfPatterns,
            Map<Integer, String> mapOfClasses) {
        this.mapOfPatterns = mapOfPatterns;
        this.mapOfClasses = mapOfClasses;

        // ============= Neighbors =====================
        this.classMemberNear = classMemberNear;

    }

    /**
     *
     * @param lk
     * <p>
     * @return gradiantOfFwrtL
     */
    public RealMatrix execute(RealMatrix lk) {

        RealMatrix metric = lk.transpose().multiply(lk);
        MetricDistance metricDistance = new MetricDistance(metric);

        // =============================================
        RealMatrix sumij = MatrixUtils.createRealMatrix(
                lk.getRowDimension(), lk.getColumnDimension());

        for (Integer idx : classMemberNear.keySet()) {

            RealVector x_i = mapOfPatterns.get(idx);
            List<Integer> listxj = classMemberNear.get(idx);

            for (Integer jdx : listxj) {
                RealVector x_j = mapOfPatterns.get(jdx);

                RealVector deltaij = x_i.subtract(x_j);
                sumij = sumij.add(deltaij.outerProduct(deltaij));

            }
        }

        sumij = lk.multiply(sumij).scalarMultiply(2.0);

        // ================================================
        RealMatrix sumijl = MatrixUtils.createRealMatrix(
                lk.getRowDimension(), lk.getColumnDimension());

        for (Integer idx : classMemberNear.keySet()) {

            RealVector x_i = mapOfPatterns.get(idx);
            List<Integer> listxj = classMemberNear.get(idx);

            for (Integer jdx : listxj) {
                RealVector x_j = mapOfPatterns.get(jdx);

                for (Integer ldx : mapOfPatterns.keySet()) {

                    if (mapOfClasses.get(ldx)
                            .equalsIgnoreCase(mapOfClasses.get(idx))) {
                        continue;
                    }

                    RealVector x_l = mapOfPatterns.get(ldx);

                    RealVector deltaij = x_i.subtract(x_j);
                    RealMatrix cij = deltaij.outerProduct(deltaij);

                    RealVector deltail = x_i.subtract(x_l);
                    RealMatrix cil = deltail.outerProduct(deltail);

                    double z = 1 + metricDistance.distance(x_i, x_j) - metricDistance.distance(x_i, x_l);

                    // εi,j,lm=1−( xi−xlm)TLTL( xi−xlm)+( xi− xj )TLTL( xi− xj ), 
                    // and if εijlm > 0, [εijlm]+ = 1, otherwise, [εijlm]+ = 0.
                    double hPrime = SupportingFunctionality.HingePrimeApproxGLL(z);

                    sumijl = sumijl.add((cij.subtract(cil)).scalarMultiply(hPrime));

                }
            }
        }

        sumijl = lk.multiply(sumijl).scalarMultiply(2.0 * GAMMA);

        return sumij.add(sumijl);
    }

    public void setGAMMA(double GAMMA) {
        this.GAMMA = GAMMA;
    }

}
