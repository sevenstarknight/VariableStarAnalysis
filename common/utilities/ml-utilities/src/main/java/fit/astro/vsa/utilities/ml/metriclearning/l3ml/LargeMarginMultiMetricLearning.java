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

import fit.astro.vsa.common.bindings.math.ml.metric.MultiViewMetric;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Hu, J., Lu, J., Yuan, J., & Tan, Y. P. (2014, November). Large margin
 * multi-metric learning for face and kinship verification in the wild. In Asian
 * Conference on Computer Vision (pp. 252-267). Springer, Cham.
 *
 * @author Kyle Johnston
 */
public class LargeMarginMultiMetricLearning {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(LargeMarginMultiMetricLearning.class);

    private final Map<Integer, Map<String, RealVector>> mapOfPatterns;
    private final Map<Integer, String> mapOfClasses;

    private final Set<String> features;

    private int MAX_ITER = Integer.MAX_VALUE;

    // ====================================================================
    // Convergence
    private double REL_ERROR = 1e-6;
    // Step Size
    private double BETA = 1e-10;
    // weight the pairwise distance between samples
    private double LAMBDA = 0.1;

    /**
     *
     * @param mapOfPatterns
     * @param mapOfClasses
     */
    public LargeMarginMultiMetricLearning(
            Map<Integer, Map<String, RealVector>> mapOfPatterns,
            Map<Integer, String> mapOfClasses) {
        this.mapOfClasses = mapOfClasses;
        this.mapOfPatterns = mapOfPatterns;
        int idx = mapOfPatterns.keySet().iterator().next();
        this.features = mapOfPatterns.get(idx).keySet();
    }

    /**
     *
     * @param tau
     * @param mu
     * @return
     */
    public Map<String, MultiViewMetric> execute(double tau, double mu) {

        Map<String, RealVector> startingSet = mapOfPatterns.values().iterator()
                .next();

        int K = startingSet.size();

        //======================================================
        // Initialize L Matrix (Covariance Matrix for) to the identity matrix
        Map<String, L3MLVariable> l3mlVariables = new HashMap<>();

        Map<String, RealMatrix> gradM_k = new HashMap<>();
        Map<String, RealMatrix> l_k = new HashMap<>();

        for (String idx : features) {
            RealMatrix lk = MatrixUtils.createRealIdentityMatrix(
                    startingSet.get(idx).getDimension()).scalarMultiply(0.001);

            gradM_k.put(idx, lk);
            l_k.put(idx, lk);

            l3mlVariables.put(idx, new L3MLVariable(lk, 1.0 / (double) K, tau, mu));
        }

        //======================================================
        // Initialization
        double jt = 0;

        LOGGER.info("Objective Function is to be minimized");
        LOGGER.info("Every Other Step Logged");

        // ================================================
        L3ML_MetricLearningObjective learningObjective
                = new L3ML_MetricLearningObjective(mapOfPatterns, mapOfClasses);
        learningObjective.setLAMBDA(LAMBDA);

        L3ML_MetricLearningGradientGenerator learningGradientGenerator
                = new L3ML_MetricLearningGradientGenerator(mapOfPatterns,
                        mapOfClasses);
        learningGradientGenerator.setLAMBDA(LAMBDA);

        for (int idx = 0; idx < MAX_ITER; idx++) {

            // ================================================
            // Step 1 Update Lk
            for (String kdx : features) {

                RealMatrix gradiantOfJwrtLMatrix = learningGradientGenerator
                        .generateLk(kdx, l3mlVariables);

                // ================================================
                double stepSize;
                if (idx == 0) {
                    stepSize = BETA;
                } else {
                    stepSize = generateBeta_BB(gradiantOfJwrtLMatrix,
                            gradM_k.get(kdx), l3mlVariables.get(kdx).getLk(), l_k.get(kdx));
                }

                RealMatrix tmpL = l3mlVariables.get(kdx).getLk()
                        .subtract(gradiantOfJwrtLMatrix.scalarMultiply(stepSize));

                // ================================================
                l_k.put(kdx, l3mlVariables.get(kdx).getLk());
                gradM_k.put(kdx, gradiantOfJwrtLMatrix);

                l3mlVariables.get(kdx).setLk(tmpL);

            }

            // Step 2 Update wk
            Map<String, Double> ikMap = learningGradientGenerator.updateWeight(l3mlVariables);

            // estimate delta opt change
            double jt_1 = 0;
            for (String kdx : features) {
                jt_1 += learningObjective.valueJK(kdx, l3mlVariables, ikMap);
            }

            double delta = Math.abs(jt_1 - jt);

            if (idx % 2 == 0) {
                LOGGER.info("Objective: " + jt_1 + "  delta:" + delta);
            }

            if (delta < REL_ERROR) {
                break;
            }

            jt = jt_1;
        }

        Map<String, MultiViewMetric> outputVar = new HashMap<>(features.size());

        features.stream().forEach((feature) -> {
            RealMatrix mk = l3mlVariables.get(feature).getLk().transpose().multiply(
                    l3mlVariables.get(feature).getLk());

            outputVar.put(feature, new MultiViewMetric(mk, l3mlVariables.get(feature).getWeight()));
        });

        return outputVar;
    }

    private double generateBeta_BB(
            RealMatrix gradiantOfJwrtLMatrix, RealMatrix gradM_k,
            RealMatrix l_k, RealMatrix l_k_1) {

        RealMatrix deltaG = gradiantOfJwrtLMatrix.subtract(gradM_k);
        RealMatrix deltaM = l_k.subtract(l_k_1);

        double top = (deltaG.multiply(deltaM.transpose()))
                .add(deltaG.transpose().multiply(deltaM)).getTrace();

        double bottom = (deltaG.multiply(deltaG.transpose())).getTrace();

        return Math.abs(top / (2 * bottom));
    }

    /**
     *
     * @param BETA
     */
    public void setBETA(double BETA) {
        this.BETA = BETA;
    }

    /**
     *
     * @param MAX_ITER
     */
    public void setMAX_ITER(int MAX_ITER) {
        this.MAX_ITER = MAX_ITER;
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
     * @param REL_ERROR
     */
    public void setREL_ERROR(double REL_ERROR) {
        this.REL_ERROR = REL_ERROR;
    }

}
