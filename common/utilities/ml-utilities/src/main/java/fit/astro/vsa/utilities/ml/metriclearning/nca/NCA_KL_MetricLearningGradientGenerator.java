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

import java.util.Map;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class NCA_KL_MetricLearningGradientGenerator {

    private final Map<String, Map<Integer, RealVector>> classMembers;
    private final Map<Integer, RealVector> setOfPatterns;
    private final Map<Integer, String> setOfClasses;

    public NCA_KL_MetricLearningGradientGenerator(
            Map<String, Map<Integer, RealVector>> classMembers,
            Map<Integer, RealVector> setOfPatterns,
            Map<Integer, String> setOfClasses) {
        this.classMembers = classMembers;
        this.setOfPatterns = setOfPatterns;
        this.setOfClasses = setOfClasses;
    }

    /**
     *
     * @param lk
     * <p>
     * @return gradiantOfFwrtL
     */
    public RealMatrix execute(RealMatrix lk) {

        RealMatrix sumOverIMatrix = MatrixUtils.createRealMatrix(
                lk.getRowDimension(),
                lk.getColumnDimension());

        for (Integer idx : setOfPatterns.keySet()) {

            RealVector x_i = setOfPatterns.get(idx);
            RealMatrix mk = (lk.transpose()).multiply(lk);

            double bottom = CycleOverAllPoints(x_i, mk);

            if (bottom == 0.0) {
                // underflow from the exp
                continue;
            }

            // p_i = sum_over_same_class(p_ij)
            double p_i = EstimateProbabilityOfCorrectClassification(idx, x_i, classMembers, mk, bottom);

            RealMatrix sumOverKMatrix = EstimateFirstTerm(x_i, mk, bottom);

            RealMatrix sumOverJMatrix = EstimateSecondTerm(idx, x_i, classMembers, mk, bottom);

            sumOverIMatrix = sumOverIMatrix.add(
                    sumOverKMatrix.subtract(sumOverJMatrix.scalarMultiply(1.0/p_i)));

        }

        /**
         * Build Gradient Matrix, Negative for Minimization (-f(A))
         * f relative to L
         */
        return lk.multiply(sumOverIMatrix).scalarMultiply(-2.0);

    }

    private double CycleOverAllPoints(RealVector x_i, RealMatrix mk) {
        // Cycle Over ALL Points in the Dataset
        double bottom = 0;
        for (RealVector x_k : setOfPatterns.values()) {

            if (!x_k.equals(x_i)) {
                RealVector deltaIK = x_i.subtract(x_k);
                double distanceSq = deltaIK.dotProduct(mk.operate(deltaIK));
                bottom = bottom + Math.exp(-distanceSq);
            }
        }

        return bottom;
    }

    /**
     * p_i = sum_over_same_class(p_ij)
     * <p>
     * @param currentPatternId
     * @param x_i
     * @param classMembers
     * @param lk
     * <p>
     * @return
     */
    private double EstimateProbabilityOfCorrectClassification(
            Integer currentPatternId,
            RealVector x_i, 
            Map<String, Map<Integer, RealVector>> classMembers,
            RealMatrix mk, double bottom) {

        double p_i = 0;

        // Cycle Over All Points IN CLASS
        Map<Integer, RealVector> classMemberList
                = classMembers.get(setOfClasses.get(currentPatternId));

        for (RealVector x_j : classMemberList.values()) {
            if (!x_j.equals(x_i)) {
                RealVector deltaIJ = x_i.subtract(x_j);
                double distanceSq = deltaIJ.dotProduct(mk.operate(deltaIJ));

                double p_ij = Math.exp(-distanceSq) / bottom;

                p_i = p_i + p_ij;
            }
        }

        return p_i;
    }

    
    private RealMatrix EstimateSecondTerm(Integer currentPatternId,
            RealVector x_i, 
            Map<String, Map<Integer, RealVector>> classMembers,
            RealMatrix mk, double bottom) {

        RealMatrix sumOverJMatrix = MatrixUtils.createRealMatrix(
                mk.getRowDimension(),
                mk.getColumnDimension());

        // Cycle Over All Points IN CLASS
        Map<Integer, RealVector> classMemberList
                = classMembers.get(setOfClasses.get(currentPatternId));

        // For Similar Classes, Loop Over Members
        for (RealVector x_j : classMemberList.values()) {
            RealVector deltaIJ = x_i.subtract(x_j);
            double distanceSq = deltaIJ.dotProduct(mk.operate(deltaIJ));

            double p_ij = Math.exp(-distanceSq) / bottom;

            // Get Outer Produce x_ik * x_ik ^ T
            RealMatrix xIJxIJT = deltaIJ.outerProduct(deltaIJ);

            // Multiple p_ik (Scalar) by Outer Product
            RealMatrix scaleOuterMatrix = xIJxIJT.scalarMultiply(p_ij);

            // Some Over J Values
            sumOverJMatrix = sumOverJMatrix.add(scaleOuterMatrix);

        }

        return sumOverJMatrix;

    }

    private RealMatrix EstimateFirstTerm(RealVector x_i, RealMatrix mk,
            double bottom) {

        RealMatrix sumOverKMatrix = MatrixUtils.createRealMatrix(
                mk.getRowDimension(),
                mk.getColumnDimension());

        for (RealVector x_k : setOfPatterns.values()) {
            if (!x_k.equals(x_i)) {
                RealVector deltaIK = x_i.subtract(x_k);
                double distanceSq = deltaIK.dotProduct(mk.operate(deltaIK));

                //Determine p_ik
                double p_ik = Math.exp(-distanceSq) / bottom;

                // Get Outer Produce x_ik * x_ik ^ T
                RealMatrix xIKxIKT = deltaIK.outerProduct(deltaIK);

                // Multiple p_ik (Scalar) by Outer Product
                RealMatrix scaleOuter = xIKxIKT.scalarMultiply(p_ik);

                // Some Over K Values
                sumOverKMatrix = sumOverKMatrix.add(scaleOuter);
            }
        }

        return sumOverKMatrix;
    }

}
