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

import fit.astro.vsa.common.datahandling.LabelHandling;
import java.util.Map;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Neighbourhood Component Analysis
 * <p>
 * Goldberger, J., Hinton, G. E., Roweis, S. T., & Salakhutdinov, R. (2004).
 * Neighbourhood components analysis. In Advances in neural information
 * processing systems (pp. 513-520).
 * <p>
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class NeighbourhoodComponentsAnalysis {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(NeighbourhoodComponentsAnalysis.class);

    private double REL_ERROR = 1e-3;

    private final Map<Integer, RealVector> mapOfPatterns;
    private final Map<Integer, String> mapOfClasses;

    private final int MAX_ITER = Integer.MAX_VALUE;

    /**
     *
     * @param mapOfPatterns map of patterns, (Index, Pattern) pair
     * @param mapOfClasses map of classes, (Index, Class) pair
     */
    public NeighbourhoodComponentsAnalysis(
            Map<Integer, RealVector> mapOfPatterns,
            Map<Integer, String> mapOfClasses) {
        // Set Up a Map of Class Types
        this.mapOfClasses = mapOfClasses;

        // Set up a Map Of Patterns
        this.mapOfPatterns = mapOfPatterns;
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
                    mapOfPatterns.values().iterator().next().getDimension()).scalarMultiply(0.01);
        } else {
            lk = MatrixUtils.createRealIdentityMatrix(
                    mapOfPatterns.values().iterator().next().getDimension()).scalarMultiply(0.01);
        }

        // ========================================
        // Set up: Split Input Patterns Into Class Groups
        Map<String, Map<Integer, RealVector>> classMembers
                = LabelHandling.sortIntoMaps(mapOfPatterns, mapOfClasses);

        // Construct Generator to Produce Gradient
        // Construct Function for Objective
        NCA_MetricLearningObjective metricLearningObjective
                = new NCA_MetricLearningObjective(classMembers,
                        mapOfPatterns, mapOfClasses);

        NCA_MetricLearningGradientGenerator ncaMetricLearningGradientGenerator
                = new NCA_MetricLearningGradientGenerator(classMembers,
                        mapOfPatterns, mapOfClasses);

        //======================================================
        // Determine the gradiant of the objective function
        double jt = metricLearningObjective.valueL(lk);
        double alphaK = 1.0, jt_1;
        
        RealMatrix grad_A_k = new Array2DRowRealMatrix(lk.getData());
        RealMatrix l_k1 = new Array2DRowRealMatrix(lk.getData());
        
        LOGGER.info("Objective Function is to be minimized");
        LOGGER.info("Every Other Step Logged");
        
        // Iterate to jt max
        for (int idx = 0; idx < MAX_ITER; idx++) {

            /**
             * Build Gradient Matrix, Direction of Maximum Increase Relative
             * Increase in f relative to L
             */
            RealMatrix gradientOfFwrtA
                    = ncaMetricLearningGradientGenerator.execute(lk);

           // ======
            if (idx != 0) {
                alphaK = generateBeta_BB(gradientOfFwrtA, grad_A_k, lk, l_k1);
            }

            // Static implmentation of beta
            grad_A_k = new Array2DRowRealMatrix(gradientOfFwrtA.getData());
            l_k1 = new Array2DRowRealMatrix(lk.getData());
            
            lk = lk.subtract(gradientOfFwrtA.scalarMultiply(alphaK));

            jt_1 = metricLearningObjective.valueL(lk);
            
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

        return (lk.transpose()).multiply(lk);
    }

    /**
     *
     * @return the M Matrix that has learned the Mahalanobis Distance
     */
    public RealMatrix generateMetric_KL() {

        // ========================================
        // Set up: Split Input Patterns Into Class Groups
        Map<String, Map<Integer, RealVector>> classMembers
                = LabelHandling.sortIntoMaps(mapOfPatterns, mapOfClasses);

        //======================================================
        // Initialize L Matrix (Covariance Matrix for) to the identity matrix
        RealMatrix lk = MatrixUtils.createRealIdentityMatrix(
                mapOfPatterns.values().iterator().next().getDimension());

        // Construct Generator to Produce Gradient
        NCA_KL_MetricLearningGradientGenerator ncaMetricLearningGradientGenerator
                = new NCA_KL_MetricLearningGradientGenerator(classMembers,
                        mapOfPatterns, mapOfClasses);

        NCA_KL_MetricLearningObjective metricLearningObjective
                = new NCA_KL_MetricLearningObjective(classMembers,
                        mapOfPatterns, mapOfClasses);

        //======================================================
        // Determine the gradiant of the objective function
        double jt = metricLearningObjective.valueL(lk);
        double alphaK = 1.0, jt_1;

        RealMatrix grad_A_k = new Array2DRowRealMatrix(lk.getData());
        RealMatrix l_k1 = new Array2DRowRealMatrix(lk.getData());
        
        LOGGER.info("Objective Function is to be minimized");
        LOGGER.info("Every Other Step Logged");
        
        for (int idx = 0; idx < MAX_ITER; idx++) {

            /**
             * Build Gradient Matrix, Direction of Minimum f relative to L
             */
            RealMatrix gradientOfFwrtA
                    = ncaMetricLearningGradientGenerator.execute(lk);

            // ======
            if (idx != 0) {
                alphaK = generateBeta_BB(gradientOfFwrtA, grad_A_k, lk, l_k1);
            }

            // Static implmentation of beta
            grad_A_k = new Array2DRowRealMatrix(gradientOfFwrtA.getData());
            l_k1 = new Array2DRowRealMatrix(lk.getData());
            
            lk = lk.subtract(gradientOfFwrtA.scalarMultiply(alphaK));

            jt_1 = metricLearningObjective.valueL(lk);

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
    
    /**
     * 
     * @param REL_ERROR 
     */
    public void setREL_ERROR(double REL_ERROR) {
        this.REL_ERROR = REL_ERROR;
    }
    
    
    
    
}
