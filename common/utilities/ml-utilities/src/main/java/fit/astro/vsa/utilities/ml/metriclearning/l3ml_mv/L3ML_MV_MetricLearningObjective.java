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
package fit.astro.vsa.utilities.ml.metriclearning.l3ml_mv;

import fit.astro.vsa.utilities.ml.MetricDistance_MV;
import fit.astro.vsa.utilities.ml.utils.SupportingFunctionality;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.apache.commons.math3.linear.RealMatrix;

/**
 *
 * @author Kyle Johnston
 */
public class L3ML_MV_MetricLearningObjective {

    //============================================================ 
    // Input
    private final Map<Integer, List<Integer>> classMemberNear;

    private final Map<Integer, Map<String, RealMatrix>> mapOfPatterns;
    private final Map<Integer, String> mapOfClasses;

    //============================================================ 
    // Internal
    private double sumij;
    private double sumijl;

    //============================================================ 
    private double LAMBDA_REG = 0.5;
    private double GAMMA_PP = 0.5;
    private double MU_CR = 0.5;
    //============================================================ 
    private double p = 2;

    /**
     *
     * @param classMemberNear
     * @param mapOfPatterns
     * @param mapOfClasses
     */
    public L3ML_MV_MetricLearningObjective(
            Map<Integer, List<Integer>> classMemberNear,
            Map<Integer, Map<String, RealMatrix>> mapOfPatterns,
            Map<Integer, String> mapOfClasses) {
        this.mapOfPatterns = mapOfPatterns;
        this.mapOfClasses = mapOfClasses;

        // ============= Neighbors =====================
        this.classMemberNear = classMemberNear;
    }

    /**
     * Find Optimizing Function
     * <p>
     * @param kdx
     * @param l3mlVariables
     * <p>
     * @return
     */
    public double valueIK(String kdx,
            Map<String, L3MLVariable_MV> l3mlVariables) {

        L3MLVariable_MV l3mlVariable = l3mlVariables.get(kdx);
        RealMatrix uk = l3mlVariable.getUk();
        RealMatrix vk = l3mlVariable.getVk();

        MetricDistance_MV metricDistanceK = new MetricDistance_MV(uk, vk);

        double scaleMN = l3mlVariable.getUk().getColumnDimension()*l3mlVariable.getVk().getColumnDimension();
        
        // ===============  Pull Error ========================
        sumij = 0;
        Map<Integer, List<Double>> mapDistancesIJ = new HashMap<>(mapOfPatterns.keySet().size());
        for (Integer idx : mapOfPatterns.keySet()) {

            RealMatrix x_i = mapOfPatterns.get(idx).get(kdx);
            mapDistancesIJ.put(idx, new ArrayList<>(classMemberNear.get(idx).size()));

            // sum_ij
            classMemberNear.get(idx).parallelStream().map((jdx)
                    -> mapOfPatterns.get(jdx).get(kdx)).map((x_j)
                    -> metricDistanceK.matrixDistance(x_i, x_j)).map((distance) -> {
                sumij = sumij + distance;
                return distance;
            }).forEachOrdered((distance) -> {
                mapDistancesIJ.get(idx).add(distance);
            });
        }

        sumij = sumij/ scaleMN;
        
        // ===============  Push Error ========================
        sumijl = 0;
        for (Integer idx : classMemberNear.keySet()) {

            RealMatrix x_i = mapOfPatterns.get(idx).get(kdx);

            // sum_ij
            for (Double distance_ij : mapDistancesIJ.get(idx)) {
                // sum_ik
                mapOfPatterns.keySet().parallelStream().filter((ldx) -> !(mapOfClasses.get(ldx)
                        .equalsIgnoreCase(mapOfClasses.get(idx)))).map((ldx)
                        -> mapOfPatterns.get(ldx).get(kdx)).map((x_l)
                        -> metricDistanceK.matrixDistance(x_i, x_l)).map((distance_il)
                        -> 1 + distance_ij/scaleMN - distance_il/scaleMN).map((z)
                        -> SupportingFunctionality.HingeApproxGLL(z)).forEachOrdered((hinge) -> {
                    // 1 - y_il
                    // Hinge Loss Function
                    sumijl = sumijl + hinge;
                });
            }
        }

        // Regularization of uk;
        double scaleMM = l3mlVariable.getUk().getColumnDimension()*l3mlVariable.getUk().getColumnDimension();
        double normUK = uk.getFrobeniusNorm()/scaleMM;
        // Regularization of vk;
        double scaleNN = l3mlVariable.getUk().getColumnDimension()*l3mlVariable.getUk().getColumnDimension();
        double normVK = vk.getFrobeniusNorm()/scaleNN;

        return GAMMA_PP * sumij + (1 - GAMMA_PP) * sumijl
                + (LAMBDA_REG) * normUK * normUK + (LAMBDA_REG) * normVK * normVK;

    }

    public double valueJK(String kdx,
            Map<String, L3MLVariable_MV> l3mlVariables, 
            Map<String, Double> ikMap) {

        L3MLVariable_MV l3mlVariable_K = l3mlVariables.get(kdx);

        double ik = Math.pow(l3mlVariable_K.getWeight(), p)
                * ikMap.get(kdx);

        sumijl = generateSumIJL(kdx, l3mlVariables);

        return ik + MU_CR * sumijl;

    }

    /**
     *
     * @param kdx
     * @param l3mlVariables
     * @return
     */
    public double valueJK(String kdx,
            Map<String, L3MLVariable_MV> l3mlVariables) {

        L3MLVariable_MV l3mlVariable_K = l3mlVariables.get(kdx);

        double ik = Math.pow(l3mlVariable_K.getWeight(), p)
                * valueIK(kdx, l3mlVariables);

        sumijl = generateSumIJL(kdx, l3mlVariables);

        return ik + MU_CR * sumijl;

    }

    private double generateSumIJL(String kdx,
            Map<String, L3MLVariable_MV> l3mlVariables) {
        L3MLVariable_MV l3mlVariable_K = l3mlVariables.get(kdx);

        RealMatrix uk = l3mlVariable_K.getUk();
        RealMatrix vk = l3mlVariable_K.getVk();

        MetricDistance_MV metricDistanceK = new MetricDistance_MV(uk, vk);

        double scaleMN = l3mlVariable_K.getUk().getColumnDimension()*l3mlVariable_K.getVk().getColumnDimension();
        
        sumijl = 0;
        for (String ldx : l3mlVariables.keySet()) {

            if (ldx.contentEquals(kdx)) {
                continue;
            }

            L3MLVariable_MV l3mlVariable_L = l3mlVariables.get(ldx);

            RealMatrix ul = l3mlVariable_L.getUk();
            RealMatrix vl = l3mlVariable_L.getVk();

            MetricDistance_MV metricDistanceL = new MetricDistance_MV(ul, vl);
            
            double scaleAB = l3mlVariable_L.getUk().getColumnDimension()*l3mlVariable_L.getVk().getColumnDimension();

            mapOfPatterns.keySet().stream().forEach((idx) -> {
                RealMatrix x_i_l = mapOfPatterns.get(idx).get(ldx);
                RealMatrix x_i_k = mapOfPatterns.get(idx).get(kdx);

                List<Integer> listxj = classMemberNear.get(idx);

                listxj.parallelStream().filter((jdx) -> !(Objects.equals(idx, jdx))).forEachOrdered((jdx) -> {
                    RealMatrix x_j_l = mapOfPatterns.get(jdx).get(ldx);
                    RealMatrix x_j_k = mapOfPatterns.get(jdx).get(kdx);

                    double d_ij_l = metricDistanceL.matrixDistance(x_i_l, x_j_l)
                            /(ul.getColumnDimension()*vl.getColumnDimension());
                    
                    double d_ij_k = metricDistanceK.matrixDistance(x_i_k, x_j_k)
                            /(uk.getColumnDimension()*vk.getColumnDimension());

                    sumijl = sumijl + (d_ij_k/scaleMN - d_ij_l/scaleAB) * (d_ij_k/scaleMN - d_ij_l/scaleAB);
                });
            });
        }

        return sumijl;
    }

    /**
     *
     * @param GAMMA_PP
     */
    public void setGAMMA_PP(double GAMMA_PP) {
        this.GAMMA_PP = GAMMA_PP;
    }

    /**
     *
     * @param LAMBDA_REG
     */
    public void setLAMBDA_REG(double LAMBDA_REG) {
        this.LAMBDA_REG = LAMBDA_REG;
    }

    /**
     *
     * @param MU_CR
     */
    public void setMU_CR(double MU_CR) {
        this.MU_CR = MU_CR;
    }

    /**
     *
     * @param p
     */
    public void setP(double p) {
        this.p = p;
    }

}
