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

import fit.astro.vsa.common.utilities.math.NumericTests;
import java.util.Map;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class NCA_MetricLearningGradientGenerator {

    private final Map<String, Map<Integer, RealVector>> classMembers;
    private final Map<Integer, RealVector> mapOfPatterns;
    private final Map<Integer, String> mapOfClasses;

    private double bottom;
    private double p_i;
    private RealMatrix sumOverIMatrix;
    private RealMatrix sumOverJMatrix;
    private RealMatrix sumOverKMatrix;

    public NCA_MetricLearningGradientGenerator(
            Map<String, Map<Integer, RealVector>> classMembers,
            Map<Integer, RealVector> mapOfPatterns,
            Map<Integer, String> mapOfClasses) {
        this.classMembers = classMembers;
        this.mapOfPatterns = mapOfPatterns;
        this.mapOfClasses = mapOfClasses;
    }

    /**
     *
     * @param lk
     * <p>
     * @return gradiantOfFwrtL
     */
    public RealMatrix execute(RealMatrix lk) {

        sumOverIMatrix = MatrixUtils.createRealMatrix(
                lk.getRowDimension(),
                lk.getColumnDimension());

        for (Integer idx : mapOfPatterns.keySet()) {

            RealVector x_i = mapOfPatterns.get(idx);
            RealMatrix mk = (lk.transpose()).multiply(lk);

            bottom = 0;
            mapOfPatterns.values().stream().filter((x_k) -> (!x_k.equals(x_i))).map((x_k)
                    -> x_i.subtract(x_k)).map((deltaIK)
                    -> deltaIK.dotProduct(mk.operate(deltaIK))).forEachOrdered((distanceSq) -> {
                bottom = bottom + Math.exp(-distanceSq);
            });

            if (NumericTests.isApproxZero(bottom)) {
                // underflow from the exp
                continue;
            }

            // p_i = sum_over_same_class(p_ij)
            EstimateProbabilityOfCorrectClassification(idx, x_i, classMembers, mk, bottom);

            EstimateFirstTerm(x_i, mk, bottom);

            EstimateSecondTerm(idx, x_i, classMembers, mk, bottom);

            sumOverIMatrix = sumOverIMatrix.add(
                    sumOverKMatrix.scalarMultiply(p_i).subtract(
                            sumOverJMatrix));

        }

        /**
         * Build Gradient Matrix, Negative for -f(A) f relative to L
         */
        return lk.multiply(sumOverIMatrix).scalarMultiply(-2.0);

    }

//    private double CycleOverAllPoints(RealVector x_i, RealMatrix mk) {
//        // Cycle Over ALL Points in the Dataset
//        double bottom = 0;
//        for (RealVector x_k : mapOfPatterns.values()) {
//
//            if (!x_k.equals(x_i)) {
//                RealVector deltaIK = x_i.subtract(x_k);
//                double distanceSq = deltaIK.dotProduct(mk.operate(deltaIK));
//                bottom = bottom + Math.exp(-distanceSq);
//            }
//        }
//
//        return bottom;
//    }
    
    private RealMatrix EstimateSecondTerm(Integer currentPatternId,
            RealVector x_i,
            Map<String, Map<Integer, RealVector>> classMembers,
            RealMatrix mk, double bottom) {

        sumOverJMatrix = MatrixUtils.createRealMatrix(
                mk.getRowDimension(),
                mk.getColumnDimension());

        // Cycle Over All Points IN CLASS
        Map<Integer, RealVector> classMemberList
                = classMembers.get(mapOfClasses.get(currentPatternId));

        // For Similar Classes, Loop Over Members
        classMemberList.values().stream().map((x_j) -> x_i.subtract(x_j)).map((deltaIJ) -> {
            double distanceSq = deltaIJ.dotProduct(mk.operate(deltaIJ));
            double p_ij = Math.exp(-distanceSq) / bottom;
            // Get Outer Produce x_ik * x_ik ^ T
            RealMatrix xIJxIJT = deltaIJ.outerProduct(deltaIJ);
            // Multiple p_ik (Scalar) by Outer Product
            return xIJxIJT.scalarMultiply(p_ij);
        }).forEachOrdered((scaleOuterMatrix) -> {
            // Some Over J Values
            sumOverJMatrix = sumOverJMatrix.add(scaleOuterMatrix);
        });

        return sumOverJMatrix;

    }

    private RealMatrix EstimateFirstTerm(RealVector x_i, RealMatrix mk,
            double bottom) {

        sumOverKMatrix = MatrixUtils.createRealMatrix(
                mk.getRowDimension(),
                mk.getColumnDimension());

        mapOfPatterns.values().stream().filter((x_k) -> (!x_k.equals(x_i)))
                .map((x_k) -> x_i.subtract(x_k)).map((deltaIK) -> {
            double distanceSq = deltaIK.dotProduct(mk.operate(deltaIK));
            //Determine p_ik
            double p_ik = Math.exp(-distanceSq) / bottom;
            // Get Outer Produce x_ik * x_ik ^ T
            RealMatrix xIKxIKT = deltaIK.outerProduct(deltaIK);
            // Multiple p_ik (Scalar) by Outer Product
            return xIKxIKT.scalarMultiply(p_ik);
        }).forEachOrdered((scaleOuter) -> {
            // Some Over K Values
            sumOverKMatrix = sumOverKMatrix.add(scaleOuter);
        });

        return sumOverKMatrix;
    }

    /**
     * p_i = sum_over_same_class(p_ij)
     * <p>
     * @param idx
     * @param x_i
     * @param classMembers
     * @param lk
     * <p>
     * @return
     */
    private double EstimateProbabilityOfCorrectClassification(
            Integer idx, RealVector x_i,
            Map<String, Map<Integer, RealVector>> classMembers,
            RealMatrix mk, double bottom) {

        p_i = 0;

        // Cycle Over All Points IN CLASS
        Map<Integer, RealVector> classMemberList
                = classMembers.get(mapOfClasses.get(idx));

        classMemberList.values().stream().filter((x_j) -> (!x_j.equals(x_i)))
                .map((x_j) -> x_i.subtract(x_j))
                .map((deltaIJ) -> deltaIJ.dotProduct(mk.operate(deltaIJ)))
                .map((distanceSq) -> Math.exp(-distanceSq) / bottom).forEachOrdered((p_ij) -> {
            p_i = p_i + p_ij;
        });

        return p_i;
    }

}
