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
package fit.astro.vsa.utilities.ml.metriclearning.ldml;

import fit.astro.vsa.common.datahandling.LabelHandling;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Guillaumin, M., Verbeek, J., & Schmid, C. (2009, September). Is that you?
 * Metric learning approaches for face identification. In Computer Vision, 2009
 * IEEE 12th international conference on (pp. 498-505). IEEE.
 *
 * Guillaumin, M., Verbeek, J., & Schmid, C. (2010, September). Multiple
 * instance metric learning from automatically labeled bags of faces. In
 * European conference on Computer Vision (pp. 634-647). Springer, Berlin,
 * Heidelberg.
 *
 * @author Kyle Johnston
 */
public class LogDiscriminantMetricLearning {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(LogDiscriminantMetricLearning.class);

    private double REL_ERROR = 1e-6;

    private final Map<Integer, RealVector> mapOfPatterns;
    private final Map<String, Map<Integer, RealVector>> classMembers;
    private Map<Integer, RealVector> ySet;

    // ===============================================
    private final int MAX_ITER = Integer.MAX_VALUE;

    /**
     *
     * @param mapOfPatterns
     * @param mapOfClasses
     */
    public LogDiscriminantMetricLearning(
            Map<Integer, RealVector> mapOfPatterns,
            Map<Integer, String> mapOfClasses) {
        this.mapOfPatterns = mapOfPatterns;

        // Set up: Split Input Patterns Into Class Groups
        this.classMembers = LabelHandling.sortIntoMaps(
                mapOfPatterns, mapOfClasses);

        initialize();
    }

    private void initialize() {

        List<String> labels = new ArrayList<>(classMembers.keySet());
        this.ySet = new HashMap<>();

        for (int idx = 0; idx < labels.size(); idx++) {

            Map<Integer, RealVector> members = classMembers
                    .get(labels.get(idx));
            RealVector yVector = new ArrayRealVector(labels.size());
            yVector.setEntry(idx, idx);

            members.keySet().forEach((jdx) -> {
                ySet.put(jdx, yVector);
            });
        }
    }

    /**
     *
     * @return
     */
    public RealMatrix generateMetric() {
        int D = mapOfPatterns.values().iterator().next().getDimension();

        return generateMetric(D);
    }

    /**
     *
     * @param intDimensions
     * <p>
     * @return the M Matrix that has learned the Mahalanobis Distance
     */
    public RealMatrix generateMetric(int intDimensions) {

        int D = mapOfPatterns.values().iterator().next().getDimension();

        RealMatrix lk;
        if (intDimensions < D) {
            lk = MatrixUtils.createRealMatrix(intDimensions,
                    mapOfPatterns.values().iterator().next().getDimension());
        } else {
            lk = MatrixUtils.createRealIdentityMatrix(
                    mapOfPatterns.values().iterator().next().getDimension());
        }

        // ========================================
        // Construct Generator to Produce Gradient
        LDML_MetricLearningObjective metricLearningObjective
                = new LDML_MetricLearningObjective(mapOfPatterns, ySet);

        LDML_MetricLearningGradientGenerator ldmlMetricLearningGradientGenerator
                = new LDML_MetricLearningGradientGenerator(mapOfPatterns, ySet);

        //======================================================
        // Determine the gradiant of the objective function
        double jt = metricLearningObjective.valueL(lk);

        double alphaK = 1.0, jt_1;

        RealMatrix grad_A_k = new Array2DRowRealMatrix(lk.getData());
        RealMatrix l_k1 = new Array2DRowRealMatrix(lk.getData());

        LOGGER.info("Objective Function is to be maximized");
        LOGGER.info("Every Other Step Logged");

        for (int idx = 0; idx < MAX_ITER; idx++) {

            /**
             * Build Gradient Matrix, Direction of Maximum Increase Relative
             * Increase in f relative to L
             */
            RealMatrix gradientOfLwrtL
                    = ldmlMetricLearningGradientGenerator.execute(lk);

            // ======
            if (idx != 0) {
                alphaK = generateBeta_BB(gradientOfLwrtL, grad_A_k, lk, l_k1);
            }

            // Static implmentation of beta
            grad_A_k = new Array2DRowRealMatrix(gradientOfLwrtL.getData());
            l_k1 = new Array2DRowRealMatrix(lk.getData());

            lk = lk.add(gradientOfLwrtL.scalarMultiply(alphaK));

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

        // Project
        return (lk.transpose()).multiply(lk);
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

    public void setREL_ERROR(double REL_ERROR) {
        this.REL_ERROR = REL_ERROR;
    }

}
