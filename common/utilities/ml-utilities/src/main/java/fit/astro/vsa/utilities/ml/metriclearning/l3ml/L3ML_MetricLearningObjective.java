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
package fit.astro.vsa.utilities.ml.metriclearning.l3ml;

import fit.astro.vsa.utilities.ml.MetricDistance;
import fit.astro.vsa.utilities.ml.utils.SupportingFunctionality;
import java.util.Map;
import java.util.Objects;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston
 */
public class L3ML_MetricLearningObjective {

    private final Map<Integer, Map<String, RealVector>> mapOfPatterns;
    private final Map<Integer, String> mapOfClasses;

    //============================================================ 
    private double iObjective;
    private double sumijl;
    //============================================================ 
    private double LAMBDA = 0.1;
    //============================================================ 
    private double p = 2;

    /**
     *
     * @param mapOfPatterns
     * @param mapOfClasses
     */
    public L3ML_MetricLearningObjective(
            Map<Integer, Map<String, RealVector>> mapOfPatterns,
            Map<Integer, String> mapOfClasses) {
        this.mapOfPatterns = mapOfPatterns;
        this.mapOfClasses = mapOfClasses;
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
            Map<String, L3MLVariable> l3mlVariables) {

        L3MLVariable l3mlVariable = l3mlVariables.get(kdx);
        RealMatrix lk = l3mlVariable.getLk();

        RealMatrix mk = (lk.transpose()).multiply(lk);
        MetricDistance metricDistanceK = new MetricDistance(mk);

        iObjective = 0;
        for (Integer idx : mapOfPatterns.keySet()) {
            RealVector x_i = mapOfPatterns.get(idx).get(kdx);

            mapOfPatterns.keySet().parallelStream().filter((jdx) -> !(Objects.equals(idx, jdx))).map((jdx) -> {
                RealVector x_j = mapOfPatterns.get(jdx).get(kdx);
                double dSquared = metricDistanceK.distance(x_i, x_j);

                double y_ij;
                if (mapOfClasses.get(jdx)
                        .equalsIgnoreCase(mapOfClasses.get(idx))) {
                    y_ij = 1;
                } else {
                    y_ij = -1;
                }

                return l3mlVariable.getTau() - y_ij * (l3mlVariable.getMu() - dSquared);
            }).map((z) -> SupportingFunctionality.HingeApproxGLL(z)).forEachOrdered((h) -> {
                iObjective = iObjective + h;
            });
        }

        return iObjective;
    }

    /**
     *
     * @param kdx
     * @param l3mlVariables
     * @param ikMap
     * @return
     */
    public double valueJK(String kdx,
            Map<String, L3MLVariable> l3mlVariables, Map<String, Double> ikMap) {

        L3MLVariable l3mlVariable_K = l3mlVariables.get(kdx);

        double ik = Math.pow(l3mlVariable_K.getWeight(), p) * ikMap.get(kdx);

        sumijl = generateSumijl(kdx, l3mlVariables);

        return ik + LAMBDA * sumijl;

    }

    /**
     *
     * @param kdx
     * @param l3mlVariables
     * @return
     */
    public double valueJK(String kdx,
            Map<String, L3MLVariable> l3mlVariables) {

        L3MLVariable l3mlVariable_K = l3mlVariables.get(kdx);

        double ik = Math.pow(l3mlVariable_K.getWeight(), p) * valueIK(kdx,
                l3mlVariables);

        sumijl = generateSumijl(kdx, l3mlVariables);

        return ik + LAMBDA * sumijl;

    }

    private double generateSumijl(String kdx,
            Map<String, L3MLVariable> l3mlVariables) {

        L3MLVariable l3mlVariable_K = l3mlVariables.get(kdx);

        RealMatrix lk = l3mlVariable_K.getLk();
        RealMatrix mk = (lk.transpose()).multiply(lk);
        MetricDistance metricDistanceK = new MetricDistance(mk);

        sumijl = 0;
        for (String ldx : l3mlVariables.keySet()) {

            if (ldx.contentEquals(kdx)) {
                continue;
            }

            L3MLVariable l3mlVariable_L = l3mlVariables.get(ldx);

            RealMatrix ll = l3mlVariable_L.getLk();
            RealMatrix ml = (ll.transpose()).multiply(ll);
            MetricDistance metricDistanceL = new MetricDistance(ml);

            mapOfPatterns.keySet().stream().forEach((idx) -> {
                RealVector x_i_l = mapOfPatterns.get(idx).get(ldx);
                RealVector x_i_k = mapOfPatterns.get(idx).get(kdx);

                mapOfPatterns.keySet().parallelStream().filter((jdx) -> !(Objects.equals(idx, jdx))).forEachOrdered((jdx) -> {
                    RealVector x_j_l = mapOfPatterns.get(jdx).get(ldx);
                    RealVector x_j_k = mapOfPatterns.get(jdx).get(kdx);

                    double d_l = metricDistanceL.distanceSqrt(x_i_l, x_j_l);
                    double d_k = metricDistanceK.distanceSqrt(x_i_k, x_j_k);

                    sumijl = sumijl + (d_k - d_l) * (d_k - d_l);
                });
            });
        }

        return sumijl;

    }

    /**
     *
     * @param LAMBDA
     */
    public void setLAMBDA(double LAMBDA) {
        this.LAMBDA = LAMBDA;
    }

    /**
     *
     * @param p
     */
    public void setP(double p) {
        this.p = p;
    }

}
