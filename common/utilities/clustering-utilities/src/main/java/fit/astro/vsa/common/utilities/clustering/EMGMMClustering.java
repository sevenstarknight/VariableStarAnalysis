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
package fit.astro.vsa.common.utilities.clustering;

import fit.astro.vsa.common.utilities.clustering.emgmm.ClusterCovariance;
import fit.astro.vsa.common.utilities.clustering.emgmm.CovarianceType;
import fit.astro.vsa.common.utilities.clustering.emgmm.ClusterGM;
import fit.astro.vsa.common.utilities.math.linearalgebra.MatrixOperations;
import fit.astro.vsa.common.utilities.math.linearalgebra.VectorOperations;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.correlation.Covariance;

/**
 * Friedman, J., Hastie, T., & Tibshirani, R. (2001). The elements of
 * statistical learning (Vol. 1, No. 10). New York, NY, USA:: Springer series in
 * statistics.
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class EMGMMClustering {

    // Initialization
    private final Map<Integer, RealVector> setOfTrainingData;
    private final int patternSize;

    //Input
    private int numClusters;

    //Internal
    private double tolerance = 0.005;
    private double incLogArrayK1, incLogArrayK0;

    //Output
    private Map<Integer, ClusterGM> clusterCenters;
    private Map<Integer, List<Integer>> clusterMembers;

    public EMGMMClustering(Map<Integer, RealVector> setOfTrainingData) {
        this.setOfTrainingData = setOfTrainingData;

        RealVector firstVector
                = setOfTrainingData.values().iterator().next();

        this.patternSize = firstVector.getDimension();
    }

    public Map<Integer, List<Integer>> execute(int numClusters) {
        return execute(numClusters, CovarianceType.HeteroscedasticUnrestricted);
    }

    public Map<Integer, List<Integer>> execute(int numClusters,
            CovarianceType covarianceType) {
        this.numClusters = numClusters;
        List<ClusterGM> clusters = 
                InitializeCluster(setOfTrainingData, numClusters);

        incLogArrayK0 = 0;
        int counter = 1;
        double delta = 1;
        while (Math.abs(delta) > tolerance) {

            // E-Step
            Map<Integer, RealVector> expectation = eStep(clusters);

            // M-Step
            mStep(clusters, expectation);

            if (counter > 100 && Math.abs(delta) < 10) {
                break;
            }


            delta = incLogArrayK1 - incLogArrayK0;
            incLogArrayK0 = incLogArrayK1;

            counter++;
        }

        Map<Integer, RealVector> expectationMap = eStep(clusters);

        //Initialize
        clusterMembers = new HashMap<>();
        for (int idx = 0; idx < numClusters; idx++) {
            clusterMembers.put(idx, new ArrayList<>());
        }

        setOfTrainingData.keySet().stream().forEach((idx) -> {
            RealVector expectation = expectationMap.get(idx);
            int clusterID = expectation.getMaxIndex();
            clusterMembers.get(clusterID).add(idx);
        });

        return clusterMembers;
    }

    private Map<Integer, RealVector> eStep(List<ClusterGM> clusters) {

        Map<Integer, RealVector> expectation = new HashMap<>();

        incLogArrayK1 = 0;
        
        for (Integer idx : setOfTrainingData.keySet()) {
            RealVector currentData = setOfTrainingData.get(idx);

            RealVector pdfCluster = new ArrayRealVector(numClusters);
            RealVector priorArray = new ArrayRealVector(numClusters);

            for (ClusterGM cluster : clusters) {

                RealVector distanceVector = currentData.subtract(cluster.getCenter());
                double distance = distanceVector.dotProduct(cluster.getInvCov().operate(distanceVector));

                pdfCluster.setEntry(cluster.getId(),
                        Math.pow(2 * Math.PI, -patternSize / 2) * cluster.getDetCov() * Math.exp(-0.5 * distance));

                priorArray.setEntry(cluster.getId(), cluster.getPrior());
            }

            double normalize = priorArray.dotProduct(pdfCluster);

            incLogArrayK1 += normalize;

            expectation.put(idx, priorArray.ebeMultiply(pdfCluster).mapDivide(normalize));
        }

        return expectation;
    }

    private void mStep(List<ClusterGM> clusters, Map<Integer, RealVector> expectationSet) {

        for (ClusterGM cluster : clusters) {

            RealVector meanCenter = new ArrayRealVector(patternSize);
            Map<Integer, Double> expectMap = new HashMap<>();
            double probSum = 0.0;
            for (Integer idx : setOfTrainingData.keySet()) {
                RealVector currentData = setOfTrainingData.get(idx);
                RealVector expectation = expectationSet.get(idx);

                double prob = expectation.getEntry(cluster.getId());

                expectMap.put(idx, prob);
                meanCenter = meanCenter.add(currentData.mapMultiply(prob));
                probSum += prob;
            }

            meanCenter = meanCenter.mapDivide(probSum);
            ClusterCovariance clusterCovariance = new ClusterCovariance(setOfTrainingData, probSum,
                    expectMap, meanCenter);
            
            RealMatrix cov = clusterCovariance.estimate(CovarianceType.HomoscedasticDiagonal);

            
            
            cluster.updateCluster(meanCenter, cov, probSum/(double)setOfTrainingData.size());
        }

    }

    public void setTolerance(double tolerance) {
        this.tolerance = tolerance;
    }

    private List<ClusterGM> InitializeCluster(Map<Integer, RealVector> setOfTrainingData,
            int numClusters) {

        RealMatrix matrix = MatrixUtils.createRealMatrix(
                setOfTrainingData.size(), patternSize);

        int counter = 0;
        for (Integer idx : setOfTrainingData.keySet()) {
            matrix.setRowVector(counter, setOfTrainingData.get(idx));
            counter++;
        }

        Covariance covariance = new Covariance(matrix);
        RealMatrix covMatrix = covariance.getCovarianceMatrix().scalarMultiply(1.0 / numClusters);

        RealVector covArray = MatrixOperations.getDiag(covMatrix);

        // dimension along which to initialize
        int indxMark = covArray.getMaxIndex();

        List<RealVector> setOfData = new ArrayList<>(setOfTrainingData.values());

        Collections.sort(setOfData, (a, b)
                -> a.getEntry(indxMark) > b.getEntry(indxMark) ? -1
                : a.getEntry(indxMark) == b.getEntry(indxMark) ? 0 : 1);

        //
        List<ClusterGM> clusters = new ArrayList<>();
        if (numClusters == 2) {
            setOfData.get(0);
            clusters.add(new ClusterGM(0, setOfData.get(0), covMatrix, numClusters));
            clusters.add(new ClusterGM(1, setOfData.get(setOfData.size() - 1), covMatrix, numClusters));
        } else {
            double[] steps = VectorOperations.linearSpace((double) setOfData.size() - 1, 
                    (double) 0.0, numClusters);
            for (int idx = 0; idx < numClusters; idx++) {
                clusters.add(new ClusterGM(idx, setOfData.get(
                        (int) Math.round(steps[idx])), 
                        covMatrix, numClusters));
            }
        }
        return clusters;
    }

    public Map<Integer, ClusterGM> getClusterCenters() {
        return clusterCenters;
    }

}
