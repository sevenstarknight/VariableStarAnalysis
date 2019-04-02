/*
 * Copyright (C) 2018 Kyle Johnston <kyjohnst2000@my.fit.edu>
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
package fit.astro.vsa.utilities.ml.metriclearning.mmc;

import fit.astro.vsa.common.utilities.math.NumericTests;
import fit.astro.vsa.utilities.ml.MetricDistance;
import java.util.Map;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class MMC_MetricLearningGradientGenerator {

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
    public MMC_MetricLearningGradientGenerator(Map<Integer, RealVector> mapOfPatterns,
            Map<Integer, String> mapOfClasses,
            Map<String, Map<Integer, RealVector>> classMembers) {
        this.mapOfPatterns = mapOfPatterns;
        this.mapOfClasses = mapOfClasses;
        this.classMembers = classMembers;
    }

    /**
     *
     * @param mk
     * <p>
     * @return gradiantOfFwrtL
     */
    public RealMatrix execute(RealMatrix mk) {

        MetricDistance metricDistance = new MetricDistance(mk);

        // ================== Push Error ===========================
        RealMatrix sumij = MatrixUtils.createRealMatrix(
                mk.getRowDimension(), mk.getColumnDimension());

        //pairwise distance
        for (Integer idx : mapOfPatterns.keySet()) {
            RealVector x_i = mapOfPatterns.get(idx);
            String label_i = mapOfClasses.get(idx);

            for (String label_j : classMembers.keySet()) {

                if (label_i.equalsIgnoreCase(label_j)) {
                    continue;
                }
                Map<Integer, RealVector> dissimilar = classMembers.get(label_j);

                for (Integer jdx : dissimilar.keySet()) {
                    RealVector x_j = dissimilar.get(jdx);
                    RealVector delta_ij = x_i.subtract(x_j);

                    double distance = metricDistance.distanceSqrt(x_i, x_j);

                    if (!NumericTests.isApproxZero(distance)) {
                        sumij = sumij.add(delta_ij.outerProduct(delta_ij));
                    }
                }
            }
        }

        return sumij;
    }

}
