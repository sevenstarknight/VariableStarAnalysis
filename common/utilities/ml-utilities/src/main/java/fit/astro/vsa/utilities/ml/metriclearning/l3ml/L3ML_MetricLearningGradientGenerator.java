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
package fit.astro.vsa.utilities.ml.metriclearning.l3ml;

import fit.astro.vsa.common.utilities.math.NumericTests;
import fit.astro.vsa.utilities.ml.MetricDistance;
import fit.astro.vsa.utilities.ml.utils.SupportingFunctionality;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston
 */
public class L3ML_MetricLearningGradientGenerator {

    //============================================================
    // Input
    private final Map<Integer, Map<String, RealVector>> mapOfPatterns;
    private final Map<Integer, String> mapOfClasses;

    //============================================================
    // Internal
    private RealMatrix sumij_Cross;
    private RealMatrix sumij;

    //============================================================ 
    private double LAMBDA = 0.1;
    private final L3ML_MetricLearningObjective l3mlObj;

    //============================================================    
    private double p = 2;

    /**
     *
     * @param mapOfPatterns
     * @param mapOfClasses
     */
    public L3ML_MetricLearningGradientGenerator(
            Map<Integer, Map<String, RealVector>> mapOfPatterns,
            Map<Integer, String> mapOfClasses) {
        this.mapOfPatterns = mapOfPatterns;
        this.mapOfClasses = mapOfClasses;

        this.l3mlObj = new L3ML_MetricLearningObjective(
                mapOfPatterns, mapOfClasses);
        l3mlObj.setLAMBDA(LAMBDA);
    }

    /**
     *
     * @param kdx
     * @param l3mlVariables
     * @return gradiantOfFwrtL
     */
    public RealMatrix generateLk(String kdx, Map<String, L3MLVariable> l3mlVariables) {

        L3MLVariable l3mlVariable_k = l3mlVariables.get(kdx);

        RealMatrix lk = l3mlVariable_k.getLk();
        RealMatrix mk = (lk.transpose()).multiply(lk);
        MetricDistance metricDistanceK = new MetricDistance(mk);

        // ===================================================
        sumij = MatrixUtils.createRealMatrix(
                mk.getRowDimension(), mk.getColumnDimension());

        mapOfPatterns.keySet().stream().forEach((idx) -> {
            RealVector x_i = mapOfPatterns.get(idx).get(kdx);

            mapOfPatterns.keySet().parallelStream().filter((jdx) -> !(Objects.equals(idx, jdx))).forEachOrdered((jdx) -> {
                RealVector x_j = mapOfPatterns.get(jdx).get(kdx);
                double dSquare = metricDistanceK.distance(x_i, x_j);

                double y_ij;
                if (mapOfClasses.get(jdx)
                        .equalsIgnoreCase(mapOfClasses.get(idx))) {
                    y_ij = 1;
                } else {
                    y_ij = -1;
                }

                double z = l3mlVariable_k.getTau() - y_ij * (l3mlVariable_k.getMu() - dSquare);
                double hPrime = SupportingFunctionality.HingePrimeApproxGLL(z);

                RealVector deltaij = x_i.subtract(x_j);
                RealMatrix cij = deltaij.outerProduct(deltaij);

                sumij = sumij.add(cij.scalarMultiply(hPrime * y_ij));
            });
        });

        sumij = sumij.scalarMultiply(Math.pow(l3mlVariable_k.getWeight(), p));

        // ==========================================================
        sumij_Cross = MatrixUtils.createRealMatrix(
                mk.getRowDimension(), mk.getColumnDimension());

        for (String ldx : l3mlVariables.keySet()) {

            if (ldx.contentEquals(kdx)) {
                continue;
            }

            L3MLVariable l3mlVariable_l = l3mlVariables.get(ldx);

            RealMatrix ll = l3mlVariable_l.getLk();
            RealMatrix ml = (ll.transpose()).multiply(ll);
            MetricDistance metricDistanceL = new MetricDistance(ml);

            mapOfPatterns.keySet().stream().forEach((idx) -> {
                RealVector x_i_l = mapOfPatterns.get(idx).get(ldx);
                RealVector x_i_k = mapOfPatterns.get(idx).get(kdx);

                mapOfPatterns.keySet().parallelStream().filter((jdx) -> !(Objects.equals(idx, jdx))).forEachOrdered((jdx) -> {
                    RealVector x_j_l = mapOfPatterns.get(jdx).get(ldx);
                    RealVector x_j_k = mapOfPatterns.get(jdx).get(kdx);

                    RealVector deltaij = x_i_k.subtract(x_j_k);
                    RealMatrix cij = deltaij.outerProduct(deltaij);

                    double d_l = metricDistanceL.distanceSqrt(x_i_l, x_j_l);
                    double d_k = metricDistanceK.distanceSqrt(x_i_k, x_j_k);
                    if (!(NumericTests.isApproxZero(d_k))) {
                        sumij_Cross = sumij_Cross.add(cij.scalarMultiply(1 - d_l / d_k));
                    }
                });
            });

        }

        sumij_Cross = sumij_Cross.scalarMultiply(LAMBDA);

        return lk.multiply(sumij.add(sumij_Cross)).scalarMultiply(2.0);
    }

    /**
     *
     * @param l3mlVariables
     * @return 
     */
    public Map<String, Double> updateWeight(
            Map<String, L3MLVariable> l3mlVariables) {

        Map<String, Double> ikMap = new HashMap<>(l3mlVariables.keySet().size()); 
        Map<String, Double> ikNumMap = new HashMap<>(l3mlVariables.keySet().size());

        double sumOverK = 0;
        for (String ldx : l3mlVariables.keySet()) {
            ikMap.put(ldx, l3mlObj.valueIK(ldx, l3mlVariables));
            double lkNum = Math.pow(1.0 / ikMap.get(ldx),  1.0 / (1.0 - p));

            ikNumMap.put(ldx, lkNum);

            sumOverK = sumOverK + lkNum;
        }

        for (String ldx : l3mlVariables.keySet()) {
            l3mlVariables.get(ldx).setWeight(ikNumMap.get(ldx) / sumOverK);
        }

        return ikMap;
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
