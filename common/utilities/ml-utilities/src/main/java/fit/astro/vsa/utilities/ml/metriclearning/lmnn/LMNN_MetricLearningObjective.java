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
package fit.astro.vsa.utilities.ml.metriclearning.lmnn;

import fit.astro.vsa.utilities.ml.MetricDistance;
import fit.astro.vsa.utilities.ml.utils.SupportingFunctionality;
import java.util.List;
import java.util.Map;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class LMNN_MetricLearningObjective {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(LMNN_MetricLearningObjective.class);
    
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
    public LMNN_MetricLearningObjective(
            Map<Integer, List<Integer>> classMemberNear,
            Map<Integer, RealVector> mapOfPatterns,
            Map<Integer, String> mapOfClasses) {
        this.mapOfPatterns = mapOfPatterns;
        this.mapOfClasses = mapOfClasses;

        // ============= Neighbors =====================
        this.classMemberNear = classMemberNear;

    }

    /**
     * Find Optimizing Function
     * <p>
     * @param lk
     * <p>
     * @return
     */
    public double valueL(RealMatrix lk) {

        //======================================================
        RealMatrix metric = lk.transpose().multiply(lk);
        MetricDistance metricDistance = new MetricDistance(metric);

        // ===============  Pull Error ========================
        double sumij = 0;

        for (Integer idx : mapOfPatterns.keySet()) {

            RealVector x_i = mapOfPatterns.get(idx);
            List<Integer> listxj = classMemberNear.get(idx);

            for (Integer jdx : listxj) {
                RealVector x_j = mapOfPatterns.get(jdx);
                sumij = sumij + metricDistance.distance(x_i, x_j);

            }
        }

        // ===============  Push Error ========================
        double sumijl = 0;

        for (Integer idx : classMemberNear.keySet()) {

            RealVector x_i = mapOfPatterns.get(idx);
            List<Integer> listxj = classMemberNear.get(idx);

            // sum_ij
            for (Integer jdx : listxj) {
                RealVector x_j = mapOfPatterns.get(jdx);

                // sum_ik
                for (Integer ldx : mapOfPatterns.keySet()) {

                    // 1 - y_il
                    if (mapOfClasses.get(ldx)
                            .equalsIgnoreCase(mapOfClasses.get(idx))) {
                        continue;
                    }

                    RealVector x_l = mapOfPatterns.get(ldx);
                    // Hinge Loss Function
                    double deltaij = metricDistance.distance(x_i, x_j);
                    double deltail = metricDistance.distance(x_i, x_l);

                    double z = 1 + deltaij - deltail;
                    
                    double hinge = SupportingFunctionality.HingeApproxGLL(z);
                    
                    sumijl = sumijl + hinge;

                }
            }
        }

        return GAMMA * sumij + (1 - GAMMA) * sumijl;

    }

    public void setGAMMA(double GAMMA) {
        this.GAMMA = GAMMA;
    }

}
