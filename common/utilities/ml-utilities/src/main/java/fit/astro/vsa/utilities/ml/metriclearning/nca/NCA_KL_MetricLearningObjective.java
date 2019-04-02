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
package fit.astro.vsa.utilities.ml.metriclearning.nca;

import fit.astro.vsa.utilities.ml.MetricDistance;
import java.util.Map;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class NCA_KL_MetricLearningObjective {

    private final Map<String, Map<Integer, RealVector>> classMembers;
    private final Map<Integer, RealVector> mapOfPatterns;
    private final Map<Integer, String> mapOfClasses;

    public NCA_KL_MetricLearningObjective(
            Map<String, Map<Integer, RealVector>> classMembers,
            Map<Integer, RealVector> mapOfPatterns,
            Map<Integer, String> mapOfClasses) {
        this.classMembers = classMembers;
        this.mapOfPatterns = mapOfPatterns;
        this.mapOfClasses = mapOfClasses;
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
        double expectation = 0;

        for (Integer idx : mapOfPatterns.keySet()) {

            RealVector x_i = mapOfPatterns.get(idx);

            //=================================================================
            // Cycle Over ALL Points in the Dataset (once per pattern)
            double bottom = 0;
            for (RealVector x_k : mapOfPatterns.values()) {

                if (!x_k.equals(x_i)) {
                    double distance = metricDistance.distance(x_i, x_k);
                    bottom = bottom + Math.exp(-distance);
                }
            }

            if (bottom == 0.0) {
                continue;
            }
            //=================================================================
            double p_i = 0;

            // Cycle Over All Points IN CLASS
            Map<Integer, RealVector> classMemberList = classMembers
                    .get(mapOfClasses.get(idx));

            for (RealVector x_j : classMemberList.values()) {
                if (!x_j.equals(x_i)) {
                    double distance = metricDistance.distance(x_i, x_j);
                    double p_ij = Math.exp(-distance) / bottom;

                    p_i = p_i + p_ij;
                }
            }

            if (p_i > 1.0) {
                p_i = 1.0;
            }

            // sum over points in training set
            expectation = expectation + Math.log(p_i);
        }

        //turn maximization into minimization (-1)
        return -expectation;
    }

}
