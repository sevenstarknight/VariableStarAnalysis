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
package fit.astro.vsa.utilities.ml.metriclearning.sj;

import fit.astro.vsa.utilities.ml.MetricDistance;
import fit.astro.vsa.utilities.ml.utils.SupportingFunctionality;
import java.util.Map;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class SJ_MetricLearningObjective {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(SJ_MetricLearningObjective.class);

    private final Map<Integer, RealVector> mapOfPatterns;
    private final Map<Integer, String> mapOfClasses;
    private final Map<String, Map<Integer, RealVector>> classMembers;

    // =======================================
    private double GAMMA = 0.5;

    /**
     *
     * @param mapOfPatterns
     * @param mapOfClasses
     * @param classMembers
     */
    public SJ_MetricLearningObjective(Map<Integer, RealVector> mapOfPatterns,
            Map<Integer, String> mapOfClasses,
            Map<String, Map<Integer, RealVector>> classMembers) {
        this.mapOfPatterns = mapOfPatterns;
        this.mapOfClasses = mapOfClasses;
        this.classMembers = classMembers;
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
        RealMatrix mk = lk.transpose().multiply(lk);

        MetricDistance metricDistance = new MetricDistance(mk);

        // ===============  Regularization ========================
        double reg = mk.getFrobeniusNorm();

        // ===============  Push Error ========================
        double sumijl = 0;

        for (Integer idx : mapOfPatterns.keySet()) {

            RealVector x_i = mapOfPatterns.get(idx);
            String label_i = mapOfClasses.get(idx);

            // Find similar
            Map<Integer, RealVector> listxj = classMembers.get(label_i);

            // sum_ij
            for (Integer jdx : listxj.keySet()) {

                //Same class
                RealVector x_j = listxj.get(jdx);

                // sum_ik
                for (Integer kdx : mapOfPatterns.keySet()) {

                    // 1 - y_il
                    if (mapOfClasses.get(kdx)
                            .equalsIgnoreCase(mapOfClasses.get(idx))) {
                        continue;
                    }

                    RealVector x_l = mapOfPatterns.get(kdx);
                    // Hinge Loss Function
                    double deltaij = metricDistance.distance(x_i, x_j);
                    double deltail = metricDistance.distance(x_i, x_l);

                    double z = 1 + deltaij - deltail;
                    
                    double hinge = SupportingFunctionality.HingeApproxGLL(z);

                    sumijl = sumijl + hinge;

                }
            }
        }

        return GAMMA * reg + (1 - GAMMA) * sumijl;

    }

    /**
     *
     * @param GAMMA
     */
    public void setGAMMA(double GAMMA) {
        this.GAMMA = GAMMA;
    }

}
