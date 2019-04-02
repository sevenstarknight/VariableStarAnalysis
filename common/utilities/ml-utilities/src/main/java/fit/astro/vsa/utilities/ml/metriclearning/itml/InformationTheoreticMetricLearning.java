/*
 * Copyright (C) 2016 Kyle Johnston
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
package fit.astro.vsa.utilities.ml.metriclearning.itml;

import fit.astro.vsa.common.utilities.math.NumericTests;
import fit.astro.vsa.utilities.ml.MetricDistance;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.PrimitiveIterator.OfInt;
import java.util.Random;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.util.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Davis, J. V., Kulis, B., Jain, P., Sra, S., & Dhillon, I. S. (2007, June).
 * Information-theoretic metric learning. In Proceedings of the 24th
 * international conference on Machine learning (pp. 209-216). ACM.
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class InformationTheoreticMetricLearning {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(InformationTheoreticMetricLearning.class);

    // ==================================
    // System Constraints
    private int maxIter = Integer.MAX_VALUE;
    private double REL_ERROR = 1e-10;

    // ==================================
    // Parameters and Settings
    private int constFactor = 20;
    private Random random = new Random();
    private RealMatrix a_0_matrix;
    private double slack = 1e-4;

    // ==================================
    // Internal Variables
    private double upperBound;
    private double lowerBound;
    private Map<Pair<Integer, Integer>, ITMLConstraint> constraints;
    private List<Pair<Integer, Integer>> setOfPairs;
    
    // ====================================
    // Inputs
    private final Map<Integer, RealVector> mapOfPatterns;
    private final Map<Integer, String> mapOfClasses;
    

    /**
     * Initialize the training data for ITML
     *
     * @param mapOfPatterns
     * @param mapOfClasses
     */
    public InformationTheoreticMetricLearning(
            Map<Integer, RealVector> mapOfPatterns,
            Map<Integer, String> mapOfClasses) {
        this.mapOfPatterns = mapOfPatterns;
        this.mapOfClasses = mapOfClasses;

        // Initialize the a_0_matrix 
        int D = mapOfPatterns.values().iterator().next().getDimension();
        this.a_0_matrix = MatrixUtils.createRealIdentityMatrix(D);

        generateUV(5, 95);

    }

    /**
     * Generate the Matrix
     *
     * @return Matrix Metric
     */
    public RealMatrix execute() {

        getConstraints();

        // ========================================
        int sizeOfConstraints = setOfPairs.size();
        List<Double> bHat = new ArrayList<>();
        List<Double> lambda = new ArrayList<>(sizeOfConstraints);
        List<Double> lambdaOld = new ArrayList<>(sizeOfConstraints);

        for (Pair<Integer, Integer> pair : setOfPairs) {
            lambda.add(0.0);
            lambdaOld.add(0.0);
            bHat.add(constraints.get(pair).getBound());
        }

        RealMatrix a_matrix = new Array2DRowRealMatrix(a_0_matrix.getData());

        // ========================================
        // Algorithm Step 3 (Repeat)
        int idx = 0, iter = 0;
        double normSum, conv;
        for (int jdx = 0; jdx < maxIter; jdx++) {

            MetricDistance distanceMetric = new MetricDistance(a_0_matrix);

            // Step 3.1 (Pick a Constraint)
            Pair<Integer, Integer> pairing = setOfPairs.get(idx);
            ITMLConstraint constraint = constraints.get(pairing);

            RealVector deltaij = constraint.getDeltaij();

            // Step 3.2 (Estimate p, the distance)
            double distance = distanceMetric.distance(deltaij);

            if(NumericTests.isApproxZero(distance) || NumericTests.isApproxZero(bHat.get(idx))){
                LOGGER.debug("Distance: " + distance);
                LOGGER.debug("bHat: " + bHat.get(idx));
            }
            
            // Generate slack
            double gammaProjected = slack / (slack + 1);

            // Step 3.3 Delta
            double adj = ((double) constraint.getY());

            // Step 3.4 Alpha
            double alpha = Math.min(lambda.get(idx),
                    adj * gammaProjected * (1 / distance - 1 / bHat.get(idx)));

            lambda.set(idx, lambda.get(idx) - alpha);

            // Step 3.5 Beta
            double beta = adj * alpha / (1 - adj * alpha * distance);

            // Step 3.6 Eta_ij update
            bHat.set(idx, 1/((1 / bHat.get(idx)) + adj*(alpha / slack)));

            // Step 3.8 Update Metric
            RealMatrix grad = a_matrix.multiply(deltaij.outerProduct(deltaij))
                    .multiply(a_matrix);

            if(Double.isNaN(grad.getEntry(0, 0))){
                LOGGER.debug("Distance: " + distance);
            }
            
            a_matrix = a_matrix.add(grad.scalarMultiply(beta));

            // =================================================
            // Test For Convergence
            if (idx == sizeOfConstraints - 1) {
                normSum = normList(lambda) + normList(lambdaOld);

                if (NumericTests.isApproxZero(normSum)) {
                    break;
                } else {
                    conv = normDelta(lambdaOld, lambda) / normSum;
                    if (conv < REL_ERROR || iter > maxIter) {
                        break;
                    }
                }

                LOGGER.info("Norm Sum: " + normSum);
                lambdaOld = new ArrayList<>(lambda);
            }

            // =================================================
            // Next Constraint
            idx = idx % (sizeOfConstraints - 1) + 1;

            // =================================================
            // Too Many Iterations
            if (jdx > maxIter - 10) {
                LOGGER.error("Convergence Time Out");
                throw new ArithmeticException("Convergence Time Out");
            }

        }

        return a_matrix;
    }

    /**
     * Find the NORM of the diff between a1 and b1
     *
     * @param a1
     * @param b1
     * @return norm
     */
    private double normDelta(List<Double> a1, List<Double> b1) {
        double sumSquare = 0;
        for (int idx = 0; idx < a1.size(); idx++) {
            double delta = a1.get(idx) - b1.get(idx);
            sumSquare += delta * delta;
        }
        return Math.sqrt(sumSquare);
    }

    /**
     * Find the Norm of a1
     *
     * @param a1
     * @return norm
     */
    private double normList(List<Double> a1) {

        double sumSquare = 0;
        sumSquare = a1.stream().map((d) -> d * d)
                .reduce(sumSquare, (accumulator, _item) -> accumulator + _item);

        return Math.sqrt(sumSquare);

    }

    /**
     * Generate Constraints Based on Training Data
     */
    private void getConstraints() {
        constraints = new HashMap<>();
        setOfPairs = new ArrayList<>();
        
        List<Integer> idxs = new ArrayList<>(mapOfPatterns.keySet());
        List<Integer> jdxs = new ArrayList<>(mapOfPatterns.keySet());

        OfInt streamIntI = random.ints(0, mapOfPatterns.size()).iterator();
        OfInt streamIntJ = random.ints(0, mapOfPatterns.size()).iterator();

        int num_constraints = constFactor * (mapOfPatterns.size()
                * (mapOfPatterns.size() - 1));

        // Generate unique constraints for 
        for (int kdx = 0; kdx < num_constraints; kdx++) {

            for (int m = 0; m < 20; m++) {
                Integer idx = idxs.get(streamIntI.next());
                Integer jdx = jdxs.get(streamIntJ.next());

                if(Objects.equals(idx, jdx)){
                    continue;
                }
                
                
                // Generate ordered pair
                Pair ijPair;
                RealVector deltaij;
                if (idx < jdx) {
                    ijPair = new Pair<>(idx, jdx);
                    deltaij = mapOfPatterns.get(idx).subtract(
                            mapOfPatterns.get(jdx));
                } else {
                    ijPair = new Pair<>(jdx, idx);
                    deltaij = mapOfPatterns.get(jdx).subtract(
                            mapOfPatterns.get(idx));
                }
                
                if(NumericTests.isApproxZero(deltaij.getNorm())){
                    continue;
                }
                
                // Test for duplication
                if (!constraints.containsKey(ijPair)) {

                    setOfPairs.add(ijPair);
                    
                    String label_i = mapOfClasses.get(idx);
                    String label_j = mapOfClasses.get(jdx);

                    if (label_i.equalsIgnoreCase(label_j)) {
                        constraints.put(ijPair, new ITMLConstraint(1, lowerBound, deltaij));
                    } else {
                        constraints.put(ijPair, new ITMLConstraint(-1, upperBound, deltaij));
                    }
                    break;
                }
            }
        }

    }

    /**
     * Computes sample histogram of the distances between rows of X and returns
     * the value of these distances at the a^th and b^th percentiles. This
     * method is used to determine the upper and lower bounds for % similarity /
     * dissimilarity constraints.
     *
     * @param athPercent [0, 100]
     * @param bthPercent [0, 100]
     */
    public final void generateUV(double athPercent, double bthPercent) {
        MetricDistance metricDistance = new MetricDistance(a_0_matrix);
        // ========================================        
        int numDist = 2000;
        double[] distances = new double[numDist];

        List<Integer> idxs = new ArrayList<>(mapOfPatterns.keySet());
        List<Integer> jdxs = new ArrayList<>(mapOfPatterns.keySet());

        OfInt streamIntI = random.ints(0, mapOfPatterns.size()).iterator();
        OfInt streamIntJ = random.ints(0, mapOfPatterns.size()).iterator();

        for (int kdx = 0; kdx < numDist; kdx++) {
            RealVector x_i = mapOfPatterns.get(idxs.get(streamIntI.next()));
            RealVector x_j = mapOfPatterns.get(jdxs.get(streamIntJ.next()));

            distances[kdx] = metricDistance.distance(x_i, x_j);
        }

        DescriptiveStatistics descriptiveStatistics
                = new DescriptiveStatistics(distances);

        // 50% percentile
        this.upperBound = descriptiveStatistics.getPercentile(bthPercent);
        this.lowerBound = descriptiveStatistics.getPercentile(athPercent);
    }

    // ==========================================================
    /**
     * Initialize MK to something other than the usual (I)
     *
     * @param a_0_matrix
     */
    public void setMk(RealMatrix a_0_matrix) {
        this.a_0_matrix = a_0_matrix;
    }

    /**
     * Default is 20
     *
     * @param constFactor
     */
    public void setConstFactor(int constFactor) {
        this.constFactor = constFactor;
    }

    /**
     * Set random generator
     *
     * @param random
     */
    public void setRandom(Random random) {
        this.random = random;
    }

    public void setSlack(double slack) {
        this.slack = slack;
    }

    

    /**
     *
     * @param maxIter
     */
    public void setMaxIter(int maxIter) {
        this.maxIter = maxIter;
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
     * @param a_0_matrix
     */
    public void setA_0_matrix(RealMatrix a_0_matrix) {
        this.a_0_matrix = a_0_matrix;
    }

}
