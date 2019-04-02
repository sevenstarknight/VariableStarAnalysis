/*
 * Copyright (C) 2018 Kyle Johnston <kyjohnst2000@my.fit.edu>
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
package fit.astro.vsa.utilities.ml.metriclearning.sj;

import fit.astro.vsa.common.datahandling.LabelHandling;
import java.util.Map;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Schultz, M., & Joachims, T. (2004). Learning a distance metric from relative
 * comparisons. In Advances in neural information processing systems (pp.
 * 41-48).
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class SJLearning {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(SJLearning.class);

    private double REL_ERROR = 1e-3;

    private final Map<Integer, RealVector> mapOfPatterns;
    private final Map<Integer, String> mapOfClasses;

    private final int MAX_ITER = Integer.MAX_VALUE;

    /**
     *
     * @param mapOfPatterns
     * @param mapOfClasses
     */
    public SJLearning(
            Map<Integer, RealVector> mapOfPatterns,
            Map<Integer, String> mapOfClasses) {
        this.mapOfClasses = mapOfClasses;
        this.mapOfPatterns = mapOfPatterns;

    }

    /**
     *
     * <p>
     * @return the M Matrix that has learned the Mahalanobis Distance
     */
    public RealMatrix generateMetric() {

        RealMatrix lk = MatrixUtils.createRealIdentityMatrix(
                mapOfPatterns.values().iterator().next().getDimension());

        Map<String, Map<Integer, RealVector>> classMembers
                = LabelHandling.sortIntoMaps(mapOfPatterns, mapOfClasses);

        // ========================================
        // Construct Generator to Produce Gradient
        SJ_MetricLearningObjective metricLearningObjective
                = new SJ_MetricLearningObjective(
                        mapOfPatterns, mapOfClasses, classMembers);

        SJ_MetricLearningGradientGenerator metricLearningGradientGenerator
                = new SJ_MetricLearningGradientGenerator(
                        mapOfPatterns, mapOfClasses, classMembers);

        //======================================================
        // Determine the gradiant of the objective function @ initial
        double jt = metricLearningObjective.valueL(lk);
        double alphaK = 1.0, jt_1;

        RealMatrix grad_A_k = new Array2DRowRealMatrix(lk.getData());
        RealMatrix l_k1 = new Array2DRowRealMatrix(lk.getData());

        LOGGER.info("Objective Function is to be minimized");
        LOGGER.info("Every Other Step Logged");

        for (int idx = 0; idx < MAX_ITER; idx++) {

            /**
             * Build Gradient Matrix, Direction of Maximum Increase Relative
             * Increase in f relative to L
             */
            RealMatrix gradientOfLwrtL
                    = metricLearningGradientGenerator.execute(lk);

            // ======
            if (idx != 0) {
                alphaK = generateBeta_BB(gradientOfLwrtL, grad_A_k, lk, l_k1);
            }

            // Static implmentation of beta
            grad_A_k = new Array2DRowRealMatrix(gradientOfLwrtL.getData());
            l_k1 = new Array2DRowRealMatrix(lk.getData());

            lk = lk.subtract(gradientOfLwrtL.scalarMultiply(alphaK));

            jt_1 = metricLearningObjective.valueL(lk);

            double delta = Math.abs(jt_1 - jt);

            if (idx % 2 == 0) {
                LOGGER.info("Objective: " + jt_1 + "  delta:" + delta);
            }

            // Update Matrix;
            if (delta < REL_ERROR) {
                break;
            }
            jt = jt_1;

        }

        return lk.transpose().multiply(lk);
    }

    private double generateBeta_BB(
            RealMatrix gradiantOfJwrtLMatrix, RealMatrix gradM_k,
            RealMatrix l_k, RealMatrix l_k_1) {

        RealMatrix deltaG = gradiantOfJwrtLMatrix.subtract(gradM_k);
        RealMatrix deltaM = l_k.subtract(l_k_1);

        double top = (deltaG.multiply(deltaM.transpose()))
                .add(deltaM.multiply(deltaG.transpose())).getTrace();

        double bottom = (deltaG.multiply(deltaG.transpose())).getTrace();

        double approx_gamma = Math.abs(top / (2 * bottom));

        return approx_gamma;

    }

    /**
     *
     * @param REL_ERROR
     */
    public void setREL_ERROR(double REL_ERROR) {
        this.REL_ERROR = REL_ERROR;
    }

}
