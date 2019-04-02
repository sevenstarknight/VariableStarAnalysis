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
package fit.astro.vsa.utilities.ml.metriclearning.mmc;

import fit.astro.vsa.common.datahandling.LabelHandling;
import fit.astro.vsa.utilities.ml.utils.SupportingFunctionality;
import java.util.Map;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optim.univariate.BrentOptimizer;
import org.apache.commons.math3.optim.univariate.SearchInterval;
import org.apache.commons.math3.optim.univariate.UnivariateObjectiveFunction;
import org.apache.commons.math3.optim.univariate.UnivariatePointValuePair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Xing, E. P., Jordan, M. I., Russell, S. J., & Ng, A. Y. (2003). Distance
 * metric learning with application to clustering with side-information. In
 * Advances in neural information processing systems (pp. 521-528).
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class MMCLearning {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(MMCLearning.class);

    private double REL_ERROR = 1e-10;

    private final Map<Integer, RealVector> mapOfPatterns;
    private final Map<Integer, String> mapOfClasses;

    private final int MAX_ITER = Integer.MAX_VALUE;

    /**
     *
     * @param mapOfPatterns
     * @param mapOfClasses
     */
    public MMCLearning(
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

        RealMatrix mk = MatrixUtils.createRealIdentityMatrix(
                mapOfPatterns.values().iterator().next().getDimension());

        Map<String, Map<Integer, RealVector>> classMembers
                = LabelHandling.sortIntoMaps(mapOfPatterns, mapOfClasses);

        // ========================================
        // Construct Generator to Produce Gradient
        MMC_MetricLearningObjective metricLearningObjective
                = new MMC_MetricLearningObjective(
                        mapOfPatterns, mapOfClasses, classMembers);

        MMC_MetricLearningGradientGenerator mmcMetricLearningGradientGenerator
                = new MMC_MetricLearningGradientGenerator(
                        mapOfPatterns, mapOfClasses, classMembers);

        //======================================================
        // Determine the gradiant of the execute function @ initial
        double jt = metricLearningObjective.execute(mk);
        boolean isDynamic = Boolean.TRUE;
        double alphaK = 1.0, jt_1;

        LOGGER.info("Objective Function is to be maximized");
        LOGGER.info("Every Other Step Logged");
        
        for (int idx = 0; idx < MAX_ITER; idx++) {

            /**
             * Build Gradient Matrix, Direction of Maximum Increase Relative
             * Increase in f relative to L
             */
            RealMatrix gradientOfLwrtL
                    = mmcMetricLearningGradientGenerator.execute(mk);

            if (isDynamic) {
                // Dynamic Estimation of Beta
                BrentOptimizer optimizer = new BrentOptimizer(REL_ERROR, 1e-8);
                UnivariatePointValuePair optimum
                        = optimizer.optimize(new UnivariateObjectiveFunction(
                                estimateBeta(metricLearningObjective, mk, gradientOfLwrtL)),
                                new MaxEval(100),
                                GoalType.MAXIMIZE,
                                new SearchInterval(0.0, 1.0));

                alphaK = optimum.getPoint();
                isDynamic = Boolean.FALSE;
            }

            mk = mk.add(gradientOfLwrtL.scalarMultiply(alphaK));

            // ==========================================
            // Constraint C1
            mk = SupportingFunctionality.ProjectSimilarity(mk, mapOfPatterns,
                    mapOfClasses, classMembers);

            // ==========================================
            // Constraint C2
            mk = SupportingFunctionality.ProjectMToPSD(mk);

            jt_1 = metricLearningObjective.execute(mk);

            if (jt_1 < jt) {
                // wrong direction, re-estimate alpha
                isDynamic = Boolean.TRUE;
            }

            double delta = Math.abs(jt_1 - jt);
            
            if(idx%2 == 0){
                LOGGER.info("Objective: " + jt_1 + "  delta:" + delta);
            }

            // Update Matrix;
            if (delta < REL_ERROR) {
                break;
            }
            jt = jt_1;

        }

        return mk;
    }

    private UnivariateFunction estimateBeta(
            MMC_MetricLearningObjective metricLearningObjective,
            RealMatrix mk,
            RealMatrix gradientOfLwrtL) {

        return (double x) -> {

            // Project into PSD Space
            RealMatrix tmpmk = mk.add(gradientOfLwrtL.scalarMultiply(x));
            tmpmk = SupportingFunctionality.ProjectMToPSD(tmpmk);

            return metricLearningObjective.execute(tmpmk);
        };
    }

    /**
     *
     * <p>
     * @return the M Matrix that has learned the Mahalanobis Distance
     */
    public RealMatrix generateMetric_Diag() {

        RealMatrix mk = MatrixUtils.createRealIdentityMatrix(
                mapOfPatterns.values().iterator().next().getDimension());

        Map<String, Map<Integer, RealVector>> classMembers
                = LabelHandling.sortIntoMaps(mapOfPatterns, mapOfClasses);

        // ========================================
        // Construct Generator to Produce Gradient
        MMC_Diag_MetricLearningObjective metricLearningObjective
                = new MMC_Diag_MetricLearningObjective(
                        mapOfPatterns, mapOfClasses, classMembers);

        MMC_Diag_MetricLearningGradientGenerator lmnnMetricLearningGradientGenerator
                = new MMC_Diag_MetricLearningGradientGenerator(
                        mapOfPatterns, mapOfClasses, classMembers);

        //======================================================
        // Determine the gradiant of the execute function @ initial
        double jt = metricLearningObjective.execute(mk);
        boolean isDynamic = Boolean.TRUE;
        double alphaK = 1.0, jt_1;
        
        LOGGER.info("Objective Function is to be maximized");
        LOGGER.info("Every Other Step Logged");
        

        for (int idx = 0; idx < MAX_ITER; idx++) {

            /**
             * Build Gradient Matrix, Direction of Maximum Increase Relative
             * Increase in f relative to L
             */
            RealMatrix gradientOfLwrtL
                    = lmnnMetricLearningGradientGenerator.execute(mk);

            if (isDynamic) {
                // Dynamic Estimation of Beta
                BrentOptimizer optimizer = new BrentOptimizer(REL_ERROR, 1e-8);
                UnivariatePointValuePair optimum
                        = optimizer.optimize(new UnivariateObjectiveFunction(
                                estimateBeta_Diag(metricLearningObjective, mk, gradientOfLwrtL)),
                                new MaxEval(100),
                                GoalType.MINIMIZE,
                                new SearchInterval(0.0, 1.0));

                alphaK = optimum.getPoint();

            }

            mk = mk.subtract(gradientOfLwrtL.scalarMultiply(alphaK));

            mk = SupportingFunctionality.ProjectMToPSD(mk);

            jt_1 = metricLearningObjective.execute(mk);

            if (jt_1 > jt) {
                isDynamic = Boolean.TRUE;
            } else {
                isDynamic = Boolean.FALSE;
            }

            double delta = Math.abs(jt_1 - jt);
            
            if(idx%2 == 0){
                LOGGER.info("Objective: " + jt_1 + "  delta:" + delta);
            }

            // Update Matrix;
            if (delta < REL_ERROR) {
                break;
            }
            jt = jt_1;

        }

        return mk;
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
     * @param metricLearningObjective
     * @param mk
     * @param gradientOfLwrtL
     * @return 
     */
    private UnivariateFunction estimateBeta_Diag(
            MMC_Diag_MetricLearningObjective metricLearningObjective,
            RealMatrix mk,
            RealMatrix gradientOfLwrtL) {

        return (double x) -> {
            RealMatrix tmpmk = mk.subtract(gradientOfLwrtL.scalarMultiply(x));
            tmpmk = SupportingFunctionality.ProjectMToPSD(tmpmk);

            return metricLearningObjective.execute(tmpmk);
        };

    }

    public void setREL_ERROR(double REL_ERROR) {
        this.REL_ERROR = REL_ERROR;
    }
    
    
}
