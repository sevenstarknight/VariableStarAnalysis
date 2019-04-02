/*
 * Copyright (C) 2018 kjohnston
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
package fit.astro.vsa.utilities.ml.training;

import fit.astro.vsa.common.datahandling.training.TrainCrossData;
import fit.astro.vsa.common.datahandling.training.TrainCrossGenerator;
import fit.astro.vsa.common.bindings.analysis.MatrixVariateTransform;
import fit.astro.vsa.common.bindings.math.Real2DCurve;
import fit.astro.vsa.common.utilities.clustering.KMeansMatrixClustering;
import fit.astro.vsa.common.utilities.clustering.data.ClusterOutput;
import fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException;
import fit.astro.vsa.common.bindings.ml.ClassificationResult;
import fit.astro.vsa.utilities.ml.knn.KNNMatrixMetric;
import fit.astro.vsa.utilities.ml.knn.KNNVectorMetric;
import fit.astro.vsa.utilities.ml.metriclearning.PushPullMatrixAnalysis;
import fit.astro.vsa.utilities.ml.performance.ClassifierPerformance;
import fit.astro.vsa.common.datahandling.training.matrix.TrainCrossMatrixData;
import fit.astro.vsa.common.datahandling.LabelHandling;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author kjohnston
 */
public class StandardTrainingImplementation {

    private final Map<Integer, String> setOfClasses;
    private final Map<String, List<Integer>> classMembers;

    private Random rand = new Random();

    private final int intClusters = 6;

    public StandardTrainingImplementation(Map<Integer, String> setOfClasses) {

        this.setOfClasses = setOfClasses;
        this.classMembers = LabelHandling.sortIntoMaps(setOfClasses);

    }

    /**
     *
     * @param numBatches
     * @return
     */
    public List<List<Integer>> generateMiniBatches(int numBatches) {
        // Initialize Batch
        List<List<Integer>> miniBatchIdx = new ArrayList<>();
        for (int idx = 0; idx < numBatches; idx++) {
            miniBatchIdx.add(idx, new ArrayList<>());
        }

        // Compile into ordered list
        List<Integer> trainingData = new ArrayList<>();
        for (String label : classMembers.keySet()) {

            List<Integer> tempList = classMembers.get(label);

            tempList.stream().forEach((idx) -> {
                trainingData.add(idx);
            });
        }

        // Randomly Permutate the TrainingSet
        Collections.shuffle(trainingData, rand);

        //=========================================================
        // batch split
        int size = trainingData.size() / numBatches;
        Iterator<Integer> iterTraining = trainingData.iterator();
        for (int i = 0; i < numBatches; i++) {
            List<Integer> idxSet = new ArrayList<>();

            int counter = 0;
            while (iterTraining.hasNext()) {
                idxSet.add(iterTraining.next());
                counter++;
                if (counter > size) {
                    break;
                }
            }
            miniBatchIdx.add(i, idxSet);
        }

        return miniBatchIdx;
    }

    /**
     * Generate Class Specific Averages
     * <p>
     * By generating means of matrices we can reduce the memory load of some of
     * the larger matrices. Provide delta values to both QDA and k-NN
     * <p>
     * Park, H., Jeon, M., and Rosen, J.B. (2003). "Lower Dimensional
     * representation of text data based on centroids and least squares", BIT
     * Numerical mathematics 43, 2, pp. 427-448
     *
     * @param inputData
     * @param trainCrossGenerator
     * @param transform
     * @return
     * @throws NotEnoughDataException
     */
    public RealVector execute_ClassDependent(
            Map<Integer, Real2DCurve> inputData,
            TrainCrossGenerator trainCrossGenerator,
            MatrixVariateTransform transform) throws NotEnoughDataException {

        // ============================================================
        // CV and Error Estimates 
        double errorProbMeanKNN = 0;

        // ============================================================
        // CV and Error Estimates 
        for (Integer jdx : trainCrossGenerator.getCrossvalMap().keySet()) {

            // ========================================================
            // Data Wrangling
            List<Integer> trainingList = trainCrossGenerator.getCrossvalMap().get(jdx);

            // Cycle Over Classes Create Mean for Class
            Map<String, RealMatrix> meanMatrixSet = new HashMap<>();
            for (String e : classMembers.keySet()) {

                // ================================================
                RealMatrix meanMatrix = MatrixUtils.createRealMatrix(
                        transform.getMatrixDimensions()[0], transform.getMatrixDimensions()[1]);

                int counter = 0;
                for (Integer idx : classMembers.get(e)) {

                    if (!trainingList.contains(idx)) {
                        continue;
                    }

                    Real2DCurve input = inputData.get(idx);

                    // Generate the MC 
                    RealMatrix matrixVar = transform.evaluate(input);

                    meanMatrix = meanMatrix.add(matrixVar);
                    counter++;
                }

                meanMatrix = meanMatrix.scalarMultiply(1.0 / (double) counter);

                // Stochastic
                for (int idx = 0; idx < transform.getMatrixDimensions()[0]; idx++) {
                    
                    
                    // Might need to remove NANs from this
                    meanMatrix.setRowVector(idx,
                            meanMatrix.getRowVector(idx).unitVector());
                    // 
//                    meanMatrix.setRow(idx,
//                            VectorOperations.unitVector(meanMatrix.getRow(idx)));
                }

                meanMatrixSet.put(e, meanMatrix);
            }

            // ============================================================
            // Cycle Over Features, Create Delta Between Mean and Feature
            Map<Integer, RealVector> featureSpace = new HashMap<>();
            for (Integer idx : inputData.keySet()) {

                Real2DCurve input = inputData.get(idx);

                // Estimate Time Domain Averages
                List<Double> pattern = new ArrayList<>();

                // Generate the VAriate 
                RealMatrix variate = transform.evaluate(input);

                for (String e : meanMatrixSet.keySet()) {

                    pattern.add(meanMatrixSet.get(e)
                            .subtract(variate)
                            .getFrobeniusNorm());
                }

                double[] arr = pattern
                        .stream()
                        .mapToDouble(Double::doubleValue)
                        .toArray();

                featureSpace.put(idx, MatrixUtils.createRealVector(arr));
            }

            TrainCrossData tcd = new TrainCrossData(
                    featureSpace, setOfClasses,
                    trainCrossGenerator.getCrossvalMap(), jdx);

            KNNVectorMetric knn = new KNNVectorMetric(
                    tcd.getSetOfTrainingPatterns(),
                    tcd.getSetOfTrainingClasses());

            ClassificationResult classificationResultKNN
                    = knn.execute(3, "Missed",
                            tcd.getSetOfCrossvalPatterns());

            double errorKNN = ClassifierPerformance.
                    estimateMisclassificationError(
                            classificationResultKNN,
                            tcd.getSetOfCrossvalClasses());

            errorProbMeanKNN += errorKNN / (double) trainCrossGenerator.getCrossvalMap().size();

        }
        return new ArrayRealVector(new double[]{errorProbMeanKNN});

    }

    /**
     * Leverage Push-Pull Classification to generate estimate based on just
     * matrix variates; generate kMean Specific Averages to use QDA
     * classification for error estimate
     * <p>
     * By generating means of matrices we can reduce the memory load of some of
     * the larger matrices
     * <p>
     * Park, H., Jeon, M., and Rosen, J.B. (2003). "Lower Dimensional
     * representation of text data based on centroids and least squares", BIT
     * Numerical mathematics 43, 2, pp. 427-448
     *
     * @param featureSpace
     * @param trainCrossGenerator
     * @return
     * @throws NotEnoughDataException
     */
    public RealVector execute_ClassIndependent(Map<Integer, RealMatrix> featureSpace,
            TrainCrossGenerator trainCrossGenerator) throws NotEnoughDataException {

        // ============================================================
        // CV and Error Estimates 
        double errorProbMeanKNN = 0;

        KMeansMatrixClustering kmmc
                = new KMeansMatrixClustering(featureSpace);

        ClusterOutput mapClusters = kmmc.execute(intClusters);

        Map<Integer, RealVector> featureSpaceVector = new HashMap<>();

        for (Integer idx : featureSpace.keySet()) {

            RealMatrix patternMatrix = featureSpace.get(idx);

            Map<Integer, RealMatrix> clusterCenters
                    = mapClusters.getClusterCenters();
            RealVector distVector = new ArrayRealVector(intClusters);

            int counter = 0;
            for (Integer jdx : clusterCenters.keySet()) {
                distVector.setEntry(counter, (clusterCenters.get(jdx)
                        .subtract(patternMatrix)).getFrobeniusNorm());
                counter++;
            }

            featureSpaceVector.put(idx, distVector);
        }

        // Cross Validation
        for (Integer idx : trainCrossGenerator.getCrossvalMap().keySet()) {

            // ========================================================
            // Data Wrangling
            TrainCrossMatrixData tcdMatrix = new TrainCrossMatrixData(featureSpace,
                    setOfClasses, trainCrossGenerator.getCrossvalMap(), idx);

            PushPullMatrixAnalysis ppml = new PushPullMatrixAnalysis(
                    tcdMatrix.getSetOfTrainingPatterns(),
                    tcdMatrix.getSetOfTrainingClasses());
            
            RealMatrix metricV = ppml.execute();
            RealMatrix metricU = MatrixUtils.createRealIdentityMatrix(ppml.getSizeMatrix()[1]);
            
            RealMatrix[] metric = new RealMatrix[]{metricU, metricV};
            
            KNNMatrixMetric knnmm = new KNNMatrixMetric(
                    tcdMatrix.getSetOfTrainingPatterns(),
                    tcdMatrix.getSetOfTrainingClasses());

            ClassificationResult classificationResultKNN
                    = knnmm.execute(3, "Missed", metric,
                            tcdMatrix.getSetOfCrossvalPatterns());
            
            double errorKNN = ClassifierPerformance.
                    estimateMisclassificationError(
                            classificationResultKNN,
                            tcdMatrix.getSetOfCrossvalClasses());

            errorProbMeanKNN += errorKNN / (double) trainCrossGenerator.getCrossvalMap().size();

            // ========================================================
        }

        return new ArrayRealVector(
                new double[]{errorProbMeanKNN});

    }

    public void setRand(Random rand) {
        this.rand = rand;
    }

}
