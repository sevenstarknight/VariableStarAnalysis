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
package fit.astro.vsa.utilities.ml.metriclearning.mmc;

import fit.astro.vsa.utilities.ml.MetricDistance;
import java.util.Map;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class MMC_Diag_MetricLearningObjective {

    private final Map<Integer, RealVector> mapOfPatterns;
    private final Map<Integer, String> mapOfClasses;
    private final Map<String, Map<Integer, RealVector>> classMembers;

    // =======================================
    /**
     *
     * @param mapOfPatterns
     * @param mapOfClasses
     * @param classMembers
     */
    public MMC_Diag_MetricLearningObjective(Map<Integer, RealVector> mapOfPatterns,
            Map<Integer, String> mapOfClasses,
            Map<String, Map<Integer, RealVector>> classMembers) {
        this.mapOfPatterns = mapOfPatterns;
        this.mapOfClasses = mapOfClasses;
        this.classMembers = classMembers;
    }

    /**
     * Find Optimizing Function
     * <p>
     * @param mk
     * <p>
     * @return
     */
    public double execute(RealMatrix mk) {

        //======================================================
        MetricDistance metricDistance = new MetricDistance(mk);

        // ===============  Pull Error ========================
        double sumij = 0;

        for (String label : classMembers.keySet()) {

            Map<Integer, RealVector> similar = classMembers.get(label);

            //pairwise distance
            for (Integer idx : similar.keySet()) {
                RealVector x_i = similar.get(idx);

                for (Integer jdx : similar.keySet()) {
                    RealVector x_j = similar.get(jdx);
                    sumij = sumij + metricDistance.distance(x_i, x_j);

                }
            }
        }
        // ===============  Push Error ========================
        double sumil = 0;

        for (Integer idx : mapOfPatterns.keySet()) {

            RealVector x_i = mapOfPatterns.get(idx);
            String label_i = mapOfClasses.get(idx);

            // sum_ij
            for (String label_j : classMembers.keySet()) {

                if (label_i.equalsIgnoreCase(label_j)) {
                    continue;
                }

                Map<Integer, RealVector> dissimilar = classMembers.get(label_j);

                // sum_ik
                for (Integer kdx : dissimilar.keySet()) {

                    RealVector x_l = mapOfPatterns.get(kdx);
                    double deltail = metricDistance.distanceSqrt(x_i, x_l);

                    sumil = sumil + deltail;
                }
            }
        }

        return sumij + Math.log(sumil);

    }

}
