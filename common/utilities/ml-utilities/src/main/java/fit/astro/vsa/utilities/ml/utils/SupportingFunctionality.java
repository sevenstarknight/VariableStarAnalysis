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
package fit.astro.vsa.utilities.ml.utils;

import fit.astro.vsa.common.bindings.math.matrix.UnivariateFunctionMapper;
import fit.astro.vsa.common.bindings.math.vector.MaxFunction;
import fit.astro.vsa.utilities.ml.MetricDistance;
import java.util.Map;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class SupportingFunctionality {

    private final static double LAMBDA = 10.0;

    /**
     * Project the metric M onto the PSD Cone Using EigenDecomposition
     *
     * @param mk
     * @return
     */
    public static RealMatrix ProjectMToPSD(RealMatrix mk) {
        EigenDecomposition eigenDecomposition = new EigenDecomposition(mk);

        RealMatrix dMatrix = eigenDecomposition.getD();
        RealMatrix vMatrix = eigenDecomposition.getV();

        dMatrix.walkInOptimizedOrder(new UnivariateFunctionMapper(
                new MaxFunction(0.0)));

        return vMatrix.multiply(dMatrix).multiply(vMatrix.transpose());
    }

    /**
     * Xing, E. P., Jordan, M. I., Russell, S. J., & Ng, A. Y. (2003). Distance
     * metric learning with application to clustering with side-information. In
     * Advances in neural information processing systems (pp. 521-528).
     *
     * @param mk the Metric
     * @param mapOfPatterns the input data features
     * @param mapOfClasses the classes for the data
     * @param classMembers the set of class members in each class type
     * @return
     */
    public static RealMatrix ProjectSimilarity(RealMatrix mk,
            Map<Integer, RealVector> mapOfPatterns,
            Map<Integer, String> mapOfClasses,
            Map<String, Map<Integer, RealVector>> classMembers) {

        MetricDistance metricDistance = new MetricDistance(mk);

        for (Integer idx : mapOfPatterns.keySet()) {

            RealVector x_i = mapOfPatterns.get(idx);
            String label_i = mapOfClasses.get(idx);

            Map<Integer, RealVector> similar = classMembers.get(label_i);

            for (Integer jdx : similar.keySet()) {

                RealVector x_j = similar.get(jdx);
                double distance = metricDistance.distance(x_i, x_j);

                while (distance > 1) {
                    mk = mk.scalarMultiply(0.01);
                    metricDistance = new MetricDistance(mk);
                    distance = metricDistance.distance(x_i, x_j);
                }
            }

        }

        return mk;
    }

    /**
     * Rennie, J.D.M. (2005) "Maximum-Margin Logistic Regression" Zhang, T., &
     * <p>
     * Oles, F. J. (2001). Text categorization based on regularized linear
     * classification methods. Information retrieval, 4(1), 5-31.
     *
     * assume LAMBDA = 3.0
     *
     * @param z
     * @return
     */
    public static double HingeApproxGLL(double z) {

        return HingeApproxGLL(z, LAMBDA);
    }

    /**
     * Rennie, J.D.M. (2005) "Maximum-Margin Logistic Regression"
     * <p>
     * Zhang, T., Oles, F. J. (2001). Text categorization based on regularized
     * linear classification methods. Information retrieval, 4(1), 5-31.
     *
     * @param z
     * @param lambda
     * @return
     */
    public static double HingeApproxGLL(double z, double lambda) {

        if (z < -10) {
            return 0.0;
        } else if (z > 10) {
            return z;
        } else {
            return (1.0 / lambda) * Math.log(1 + Math.exp((z) * lambda));
        }
    }

    /**
     * Rennie, J.D.M. (2005) "Maximum-Margin Logistic Regression" Zhang, T., &
     * <p>
     * Oles, F. J. (2001). Text categorization based on regularized linear
     * classification methods. Information retrieval, 4(1), 5-31.
     *
     * assume LAMBDA = 3.0
     *
     * @param z
     * @return
     */
    public static double HingePrimeApproxGLL(double z) {

        return HingePrimeApproxGLL(z, LAMBDA);
    }

    /**
     * Rennie, J.D.M. (2005) "Maximum-Margin Logistic Regression" Zhang, T., &
     * Oles, F. J. (2001). Text categorization based on regularized linear
     * classification methods. Information retrieval, 4(1), 5-31.
     *
     * @param z
     * @param lambda
     * @return
     */
    public static double HingePrimeApproxGLL(double z, double lambda) {
         
        if (z < -10) {
            return 0.0;
        } else if (z > 10) {
            return 1.0;
        } else {
            return (Math.exp(z * lambda) 
                    / (1 + Math.exp(z * lambda)));
        }
    }

    /**
     *
     * @param betaParams
     * @param currentPattern
     * @return
     */
    public static RealVector SoftMax(
            Map<String, RealVector> betaParams,
            RealVector currentPattern) {

        // Estimate Softmax
        RealVector softMaxProb = new ArrayRealVector(betaParams.keySet().size());

        int counter = 0;
        for (String classType : betaParams.keySet()) {

            RealVector beta_i = betaParams.get(classType);

            double numerator = Math.exp(beta_i.dotProduct(currentPattern));

            double denominator = 0;
            for (String uniqueClassb : betaParams.keySet()) {
                RealVector w_j = betaParams.get(uniqueClassb);
                double b = w_j.dotProduct(currentPattern);
                denominator += Math.exp(b);
            }

            // Numerical Error Problems Handled Here
            if (Double.isInfinite(numerator) || Double.isNaN(numerator)) {
                softMaxProb.setEntry(counter, 1.0);
            } else {
                double tmp = Math.log(numerator) - Math.log(denominator);

                if (tmp < -50) {
                    softMaxProb.setEntry(counter, 0.0);
                } else {
                    softMaxProb.setEntry(counter, Math.exp(tmp));
                }
            }

            counter++;
        }

        return softMaxProb;
    }
}
