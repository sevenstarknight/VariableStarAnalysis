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
package fit.astro.vsa.utilities.ml.ecva;

import fit.astro.vsa.common.utilities.math.linearalgebra.MatrixOperations;
import fit.astro.vsa.common.utilities.math.linearalgebra.VectorOperations;
import fit.astro.vsa.common.datahandling.training.TrainCrossData;
import fit.astro.vsa.common.datahandling.training.TrainCrossTestGenerator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * Nørgaard, L., Bro, R., Westad, F., & Engelsen, S. B. (2006). A modification
 * of canonical variates analysis to handle highly collinear multivariate data.
 * Journal of Chemometrics: A Journal of the Chemometrics Society, 20(8‐10),
 * 425-435.
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class ECVA {

    private Random RAND = new Random();

    private final Map<Integer, RealVector> mapOfPatterns;
    private final Map<Integer, String> mapOfClasses;

    private int no_of_comp = 20;

    /**
     *
     * @param mapOfPatterns
     * @param mapOfClasses
     */
    public ECVA(Map<Integer, RealVector> mapOfPatterns,
            Map<Integer, String> mapOfClasses) {
        this.mapOfPatterns = mapOfPatterns;
        this.mapOfClasses = mapOfClasses;
    }

    /**
     *
     * @return
     */
    public CanonicalVariates execute() {

        int dimension = mapOfPatterns.values()
                .iterator().next().getDimension();

        // =========================================
        // Raw Data Matrix
        RealMatrix x = MatrixOperations.generateMatrixFromMap(
                mapOfPatterns);

        RealVector mx = new ArrayRealVector(dimension);

        RealVector stdx = new ArrayRealVector(dimension);

        for (int idx = 0; idx < dimension; idx++) {
            RealVector columnVector = x.getColumnVector(idx);
            mx.setEntry(idx, VectorOperations.mean(columnVector));
            stdx.setEntry(idx, VectorOperations.std(columnVector));
        }

        // =========================================
        TrainCrossTestGenerator trainTest = new TrainCrossTestGenerator(
                mapOfClasses, 0.2, RAND);

        // =========================================
        // 
        int minNumberOfComponents, maxNumberOfComponents;

        if (trainTest.getClassMembers().size() > mapOfPatterns.size()) {
            minNumberOfComponents = dimension;
            maxNumberOfComponents = minNumberOfComponents;
        } else {
            minNumberOfComponents = trainTest.getClassMembers().size();

            if (no_of_comp > dimension) {
                maxNumberOfComponents = dimension;
            } else {
                maxNumberOfComponents = no_of_comp;
            }

        }

        // =========================================
        // TODO Allow for Input Prior Prob
        Map<String, Double> priorProbMap = new HashMap<>(
                trainTest.getClassMembers().keySet().size());
        for (String classLabel : trainTest.getClassMembers().keySet()) {
            priorProbMap.put(classLabel,
                    1.0 / (double) trainTest.getClassMembers().size());
        }

        // ============================================
        // CV 5-Fold 
        Map<Integer, List<Integer>> crossvalMap = TrainCrossTestGenerator
                .generateCrossVal123(mapOfClasses, 5);

        Map<Integer, Map<Integer, RealVector>> predScores = new HashMap<>(
                maxNumberOfComponents - minNumberOfComponents + 1);

        Map<Integer, Map<Integer, RealVector>> posteriorMapCross = new HashMap<>(
                maxNumberOfComponents - minNumberOfComponents + 1);
        Map<Integer, Map<Integer, String>> classEstimateMapCross = new HashMap<>(
                maxNumberOfComponents - minNumberOfComponents + 1);

        for (int jdx = minNumberOfComponents - 1; jdx < maxNumberOfComponents; jdx++) {
            posteriorMapCross.put(jdx, new HashMap<>());
            classEstimateMapCross.put(jdx, new HashMap<>());
        }

        // Loop Over Folds
        int segments = 5;

        for (int idx = 0; idx < segments; idx++) {

            TrainCrossData crossData = new TrainCrossData(
                    mapOfPatterns, mapOfClasses, crossvalMap, idx);

            SubECVA secvaCross = new SubECVA(
                    crossData.getSetOfTrainingPatterns(),
                    crossData.getSetOfTrainingClasses(),
                    minNumberOfComponents, maxNumberOfComponents);

            // Adjust crossval data
            RealMatrix xTraining = MatrixOperations.generateMatrixFromMap(
                    crossData.getSetOfTrainingPatterns());

            RealVector sumOfColumns
                    = MatrixOperations.dimensionalSummation(xTraining, Boolean.TRUE);

            RealVector meanOfColumns
                    = sumOfColumns.mapDivide((double) xTraining.getRowDimension());

            // Estimate Values
            for (int jdx = minNumberOfComponents - 1; jdx < maxNumberOfComponents; jdx++) {

                Map<Integer, RealVector> mapOfScores = new HashMap<>();
                for (Integer kdx : crossData.getSetOfCrossvalPatterns().keySet()) {
                    RealMatrix tmpMatrix = MatrixUtils.createRowRealMatrix(
                            crossData.getSetOfCrossvalPatterns().get(kdx)
                                    .subtract(meanOfColumns).toArray());
                    mapOfScores.put(kdx,
                            tmpMatrix.multiply(secvaCross.getCvaWeights()
                                    .get(jdx)).getRowVector(0));

                }

                predScores.put(jdx, mapOfScores);
            }

            for (int jdx = minNumberOfComponents - 1; jdx < maxNumberOfComponents; jdx++) {
                SubLDA slda = new SubLDA(predScores.get(jdx),
                        secvaCross.getProjScores().get(jdx), mapOfClasses, priorProbMap);

                posteriorMapCross.get(jdx).putAll(slda.getPosteriorMap());
                classEstimateMapCross.get(jdx).putAll(slda.getClassEstimateMap());
            }
        }

        // ====================================================================
        // Estimate Misclassification Rate
        Map<Integer, Double> misclassification = new HashMap<>(maxNumberOfComponents - minNumberOfComponents + 1);
        double minRate = 1.0;
        Integer optComp = 0;
        for (int jdx = minNumberOfComponents - 1; jdx < maxNumberOfComponents; jdx++) {
            Map<Integer, String> tmpMap = classEstimateMapCross.get(jdx);
            double missclass = 0.0;
            missclass = tmpMap.keySet().stream().filter((idx)
                    -> (!tmpMap.get(idx).equalsIgnoreCase(mapOfClasses.get(idx)))).map((_item) -> 1.0)
                    .reduce(missclass, (accumulator, _item) -> accumulator + 1);

            double misclassrate = missclass / (double) tmpMap.entrySet().size();

            if (minRate > misclassrate) {
                optComp = jdx;
                minRate = misclassrate;
            }

            misclassification.put(jdx, misclassrate);
        }

        //====================================================================
        // Apply ECVA to whole dataset
        Map<Integer, RealVector> posteriorMap = new HashMap<>(
                maxNumberOfComponents - minNumberOfComponents + 1);
        Map<Integer, String> classEstimateMap = new HashMap<>(
                maxNumberOfComponents - minNumberOfComponents + 1);

        posteriorMap.putAll(posteriorMapCross.get(optComp));
        classEstimateMap.putAll(classEstimateMapCross.get(optComp));

        SubECVA secva = new SubECVA(mapOfPatterns,
                mapOfClasses, minNumberOfComponents, maxNumberOfComponents);

        //====================================================================
        return new CanonicalVariates(mapOfPatterns, mapOfClasses,
                secva.getProjScores().get(optComp), mx, stdx,
                secva.getCvaWeights().get(optComp));

    }

    public void setNo_of_comp(int no_of_comp) {
        this.no_of_comp = no_of_comp;
    }

    public void setRAND(Random RAND) {
        this.RAND = RAND;
    }

}
