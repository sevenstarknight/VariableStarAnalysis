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
package fit.astro.vsa.utilities.ml.metriclearning.pmml;

import fit.astro.vsa.common.bindings.math.ml.metric.MultiViewMetric;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.solvers.BrentSolver;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.CombinatoricsUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Cui, Z., Li, W., Xu, D., Shan, S., & Chen, X. (2013). Fusing robust face
 * region descriptors via multiple metric learning for face recognition in the
 * wild. In Proceedings of the IEEE Conference on Computer Vision and Pattern
 * Recognition (pp. 3554-3561).
 *
 * @author Kyle Johnston
 */
public class PairwiseMultipleMetricLearning {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(PairwiseMultipleMetricLearning.class);

    private final Map<Integer, Map<String, RealVector>> mapOfPatterns;
    private final Map<Integer, String> mapOfClasses;

    private final Set<String> features;

    private Random rand = new Random();
    private int MAX_ITER = Integer.MAX_VALUE;

    // ====================================================================
    // Convergence
    private double REL_ERROR = 1e-7;

    /**
     *
     * @param mapOfPatterns
     * @param mapOfClasses
     */
    public PairwiseMultipleMetricLearning(
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
     * @param rho
     * @param gamma
     * @return
     */
    public Map<String, MultiViewMetric> execute(double tau, double rho, double gamma) {

        Map<String, RealVector> startingSet = mapOfPatterns.values().iterator()
                .next();

        //======================================================
        // Initialize L Matrix (Covariance Matrix for) to the identity matrix
        Map<String, MultiViewMetric> pmmlVariables = new HashMap<>();

        for (String idx : features) {
            RealMatrix mk = MatrixUtils.createRealIdentityMatrix(
                    startingSet.get(idx).getDimension()).scalarMultiply(0.001);

            pmmlVariables.put(idx, new MultiViewMetric(mk, 1.0 / (double) features.size()));
        }

        //
        List<Integer> indexValues = new ArrayList<>(mapOfPatterns.keySet());
        Iterator<int[]> comboIter = CombinatoricsUtils
                .combinationsIterator(indexValues.size(), 2);

        //
        Map<int[], Double> eta_ijs = new HashMap<>();
        Map<int[], Double> nu_ijs = new HashMap<>();
        Map<int[], Double> del_ijs = new HashMap<>();

        while (comboIter.hasNext()) {
            int[] arrayOfIdx = comboIter.next();

            double del_ij;
            if (mapOfClasses.get(indexValues.get(arrayOfIdx[0])).equalsIgnoreCase(
                    mapOfClasses.get(indexValues.get(arrayOfIdx[1])))) {
                del_ij = 1.0;
            } else {
                del_ij = -1.0;
            }
            // fixed
            del_ijs.put(arrayOfIdx, del_ij);
            // initialization
            eta_ijs.put(arrayOfIdx, del_ij * rho - tau);
            // initialization
            nu_ijs.put(arrayOfIdx, 0.0);
        }

        PMML_MetricLearningObjective learningObjective
                = new PMML_MetricLearningObjective(del_ijs);
        learningObjective.setGamma(gamma);
        learningObjective.setRho(rho);
        learningObjective.setTau(tau);

        //======================================================
        // Initialization
        double jt = 0;

        LOGGER.info("Objective Function is to be minimized");
        LOGGER.info("Every Other Step Logged");

        List<int[]> keySet = new ArrayList<>(del_ijs.keySet());

        double n2gamma = (double) mapOfPatterns.size() / (2.0 * gamma);

        for (int idx = 0; idx < MAX_ITER; idx++) {

            // ================================================
            // Step 1 Random Selection of ij
            int[] randKey_i = keySet.get(rand.nextInt(keySet.size()));

            Map<String, RealVector> x_i = mapOfPatterns.get(indexValues.get(randKey_i[0]));
            Map<String, RealVector> x_j = mapOfPatterns.get(indexValues.get(randKey_i[1]));

            if (!testMargin(pmmlVariables, x_i, x_j,
                    del_ijs.get(randKey_i), tau, rho)) {
                continue;
            }

            // ================================================
            // Step 2 Dynamic Estimation of Beta
            AlphaFunction af = new AlphaFunction(pmmlVariables, x_i, x_j,
                    eta_ijs.get(randKey_i),
                    del_ijs.get(randKey_i), n2gamma);

            BrentSolver optimizer = new BrentSolver(REL_ERROR, 1e-8);
            double alphaK = optimizer.solve(100,
                    af, -100.0, 100.0);

            alphaK = Math.min(alphaK, nu_ijs.get(randKey_i));

            double updateNu = nu_ijs.get(randKey_i) - alphaK;
            nu_ijs.put(randKey_i, updateNu);

            // ================================================
            // Step 3 Update Lk
            for (String kdx : features) {

                RealVector deltaij = x_i.get(kdx).subtract(x_j.get(kdx));

                RealMatrix mk = pmmlVariables.get(kdx).getMk();
                double dij = deltaij.dotProduct(mk.operate(deltaij));
                double del = del_ijs.get(randKey_i);

                double mu = (del * alphaK) / (1 - del * alphaK * dij);

                RealMatrix mk_t = pmmlVariables.get(kdx).getMk();

                RealMatrix mk_t1 = mk_t.add((mk_t.multiply(deltaij.outerProduct(deltaij))
                        .multiply(mk_t))
                        .scalarMultiply(mu));

                pmmlVariables.put(kdx, new MultiViewMetric(mk_t1, 1.0 / (double) features.size()));
            }

            // ================================================
            // Step 4 Update eta
            double etaTmp = eta_ijs.get(randKey_i) - n2gamma * alphaK;
            eta_ijs.put(randKey_i, etaTmp);
            
            // ================================================
            // estimate delta opt change
            double jt_1 = learningObjective.estimateObjective(eta_ijs, pmmlVariables);

            double delta = Math.abs(jt_1 - jt);

            if (idx % 2 == 0) {
                LOGGER.info("Objective: " + jt_1 + "  delta:" + delta);
            }

            if (delta < REL_ERROR) {
                break;
            }

            jt = jt_1;
        }

        return pmmlVariables;
    }

    /**
     *
     * @param rand
     */
    public void setRand(Random rand) {
        this.rand = rand;
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
     * @param REL_ERROR
     */
    public void setREL_ERROR(double REL_ERROR) {
        this.REL_ERROR = REL_ERROR;
    }

    /**
     *
     */
    private static class AlphaFunction implements UnivariateFunction {

        private final Map<String, MultiViewMetric> metrics;
        private final Map<String, RealVector> x_i;
        private final Map<String, RealVector> x_j;

        private final double eta_ij;
        private final double del_ij;
        private final double n2gamma;

        /**
         *
         * @param metrics
         * @param x_i
         * @param x_j
         * @param eta_ij
         * @param del_ij
         * @param n2gamma
         */
        public AlphaFunction(Map<String, MultiViewMetric> metrics,
                Map<String, RealVector> x_i, Map<String, RealVector> x_j,
                double eta_ij, double del_ij, double n2gamma) {
            this.metrics = metrics;
            this.x_i = x_i;
            this.x_j = x_j;
            this.eta_ij = eta_ij;
            this.del_ij = del_ij;
            this.n2gamma = n2gamma;
        }

        @Override
        public double value(double alpha) {

            double sum = 0;

            for (String view : metrics.keySet()) {
                RealVector deltaij = x_i.get(view).subtract(x_j.get(view));
                RealMatrix mk = metrics.get(view).getMk();
                double dij = deltaij.dotProduct(mk.operate(deltaij));

                sum += dij / (1.0 - del_ij * alpha * dij);
            }

            return (del_ij * sum) / (double) metrics.keySet().size() - (eta_ij - n2gamma * alpha);
        }

    }

    private boolean testMargin(Map<String, MultiViewMetric> metrics,
            Map<String, RealVector> x_i, Map<String, RealVector> x_j,
            double del_ij, double tau, double rho) {

        double sum = 0;
        for (String view : metrics.keySet()) {
            RealVector deltaij = x_i.get(view).subtract(x_j.get(view));
            RealMatrix mk = metrics.get(view).getMk();
            double dij = deltaij.dotProduct(mk.operate(deltaij));

            sum += dij;
        }
        return ((del_ij * sum) / (double) metrics.keySet().size()) > (del_ij * rho - tau);
    }

}
