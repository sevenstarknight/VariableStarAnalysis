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
package fit.astro.vsa.utilities.ml.metriclearning.lmnn_mv;

import fit.astro.vsa.utilities.ml.MetricDistance_MV;
import fit.astro.vsa.utilities.ml.utils.SupportingFunctionality;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.commons.math3.linear.RealMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class LMNN_MV_MetricLearningObjective {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(LMNN_MV_MetricLearningObjective.class);

    private final Map<Integer, RealMatrix> mapOfPatterns;
    private final Map<Integer, String> mapOfClasses;

    //============================================================ 
    // Internal
    private double sumij;
    private double sumijl;

    // =======================================
    private final Map<Integer, List<Integer>> classMemberNear;

    private double GAMMA = 0.5;
    private double LAMBDA = 0.5;

    /**
     *
     * @param classMemberNear
     * @param mapOfPatterns
     * @param mapOfClasses
     */
    public LMNN_MV_MetricLearningObjective(
            Map<Integer, List<Integer>> classMemberNear,
            Map<Integer, RealMatrix> mapOfPatterns,
            Map<Integer, String> mapOfClasses) {
        this.mapOfPatterns = mapOfPatterns;
        this.mapOfClasses = mapOfClasses;

        // ============= Neighbors =====================
        this.classMemberNear = classMemberNear;

    }

    /**
     * Find Optimizing Function
     * <p>
     * @param gammak
     * @param nuk
     * <p>
     * @return
     */
    public double valueL(RealMatrix gammak, RealMatrix nuk) {

        //======================================================
        RealMatrix uk = gammak.transpose().multiply(gammak);
        RealMatrix vk = nuk.transpose().multiply(nuk);

        MetricDistance_MV metricDistance = new MetricDistance_MV(uk, vk);

        // ===============  Pull Error ========================
        sumij = 0;
        Map<Integer, List<Double>> mapDistancesIJ = new HashMap<>(mapOfPatterns.keySet().size());
        for (Integer idx : mapOfPatterns.keySet()) {

            RealMatrix x_i = mapOfPatterns.get(idx);
            mapDistancesIJ.put(idx, new ArrayList<>(classMemberNear.get(idx).size()));

            // sum_ij
            classMemberNear.get(idx).parallelStream().map((jdx)
                    -> mapOfPatterns.get(jdx)).map((x_j)
                    -> metricDistance.matrixDistance(x_i, x_j)).map((distance) -> {
                sumij = sumij + distance;
                return distance;
            }).forEachOrdered((distance) -> {
                mapDistancesIJ.get(idx).add(distance);
            });
        }

        // ===============  Push Error ========================
        // ===============  Push Error ========================
        sumijl = 0;
        for (Integer idx : classMemberNear.keySet()) {

            RealMatrix x_i = mapOfPatterns.get(idx);

            // sum_ij
            for (Double distance_ij : mapDistancesIJ.get(idx)) {
                // sum_ik
                mapOfPatterns.keySet().parallelStream().filter((ldx) -> !(mapOfClasses.get(ldx)
                        .equalsIgnoreCase(mapOfClasses.get(idx)))).map((ldx)
                        -> mapOfPatterns.get(ldx)).map((x_l)
                        -> metricDistance.matrixDistance(x_i, x_l)).map((distance_il)
                        -> 1 + distance_ij - distance_il).map((z)
                        -> SupportingFunctionality.HingeApproxGLL(z)).forEachOrdered((hinge) -> {
                    // 1 - y_il
                    // Hinge Loss Function
                    sumijl = sumijl + hinge;
                });
            }
        }

        // Regularization of uk;
        double normUK = uk.getFrobeniusNorm();
        // Regularization of vk;
        double normVK = vk.getFrobeniusNorm();

        return GAMMA * sumij + (1 - GAMMA) * sumijl
                + (LAMBDA) * normUK * normUK + (LAMBDA) * normVK * normVK;

    }

    /**
     * 
     * @param GAMMA 
     */
    public void setGAMMA(double GAMMA) {
        this.GAMMA = GAMMA;
    }

    /**
     * 
     * @param LAMBDA 
     */
    public void setLAMBDA(double LAMBDA) {
        this.LAMBDA = LAMBDA;
    }

}
