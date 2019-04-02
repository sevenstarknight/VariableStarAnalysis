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
package fit.astro.vsa.utilities.ml.metriclearning.lmnn_mv;

import fit.astro.vsa.common.utilities.math.support.SortingOperations;
import fit.astro.vsa.utilities.ml.MetricDistance_MV;
import fit.astro.vsa.common.datahandling.LabelHandling;
import fit.astro.vsa.common.utilities.math.linearalgebra.MatrixOperations;
import fit.astro.vsa.utilities.ml.ecva.CanonicalVariates;
import fit.astro.vsa.utilities.ml.ecva.ECVA;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Weinberger, K. Q., Blitzer, J., & Saul, L. K. (2006). Distance metric
 * learning for large margin nearest neighbor classification. In Advances in
 * neural information processing systems (pp. 1473-1480).
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class LargeMarginNearestNeighbor_MV {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(LargeMarginNearestNeighbor_MV.class);

    private double REL_ERROR = 1e-6;

    private final Map<Integer, RealMatrix> mapOfPatterns;
    private final Map<Integer, String> mapOfClasses;

    private Map<Integer, List<Integer>> classMemberNear;
    // ===============================================
    private final int MAX_ITER = Integer.MAX_VALUE;

    /**
     *
     * @param mapOfPatterns
     * @param mapOfClasses
     * @param kValue
     */
    public LargeMarginNearestNeighbor_MV(
            Map<Integer, RealMatrix> mapOfPatterns,
            Map<Integer, String> mapOfClasses, int kValue) {
        this.mapOfClasses = mapOfClasses;
        this.mapOfPatterns = mapOfPatterns;

        // ========================================
        // Set up: Split Input Patterns Into Class Groups
        initializeNeighbors(kValue);
    }

    /**
     *
     * @return
     */
    public RealMatrix[] generateMetric() {
        RealMatrix tmpMatrix = mapOfPatterns.values().iterator()
                .next();

        return generateMetric(tmpMatrix.getRowDimension(), tmpMatrix.getColumnDimension());
    }

    /**
     *
     * @param intRow
     * @param intColumn
     * @return the M Matrix that has learned the Mahalanobis Distance
     */
    public RealMatrix[] generateMetric(int intRow, int intColumn) {

        RealMatrix gammak = MatrixUtils.createRealIdentityMatrix(intColumn);

        RealMatrix nuk = MatrixUtils.createRealIdentityMatrix(intRow);

        // ========================================
        // Construct Generator to Produce Gradient
        LMNN_MV_MetricLearningObjective metricLearningObjective
                = new LMNN_MV_MetricLearningObjective(
                        classMemberNear, mapOfPatterns, mapOfClasses);

        LMNN_MV_MetricLearningGradientGenerator lmnnMetricLearningGradientGenerator
                = new LMNN_MV_MetricLearningGradientGenerator(
                        classMemberNear, mapOfPatterns, mapOfClasses);

        //======================================================
        // Determine the gradiant of the objective function @ initial
        double jt = metricLearningObjective.valueL(gammak, nuk);
        double step_u = 1.0, step_v = 1.0, jt_1;

        RealMatrix gradU_k = new Array2DRowRealMatrix(gammak.getData());
        RealMatrix gradV_k = new Array2DRowRealMatrix(nuk.getData());

        RealMatrix g_k = new Array2DRowRealMatrix(gammak.getData());
        RealMatrix n_k = new Array2DRowRealMatrix(nuk.getData());

        LOGGER.info("Objective Function is to be minimized");
        LOGGER.info("Every Other Step Logged");

        for (int idx = 0; idx < MAX_ITER; idx++) {

            /**
             * Build Gradient Matrix, Direction of Maximum Increase Relative
             * Increase in f relative to L
             */
            RealMatrix[] gradientOfLwrtL
                    = lmnnMetricLearningGradientGenerator.execute(gammak, nuk);

            // ======
            if (idx != 0) {
                step_u = generateBeta_BB(gradientOfLwrtL[0],
                        gradU_k, gammak, g_k);

                step_v = generateBeta_BB(gradientOfLwrtL[1],
                        gradV_k, nuk, n_k);
            }
            // Static implmentation of beta
            gammak = gammak.subtract(gradientOfLwrtL[0].scalarMultiply(step_u));

            nuk = nuk.subtract(gradientOfLwrtL[1].scalarMultiply(step_v));

            jt_1 = metricLearningObjective.valueL(gammak, nuk);

            double delta = Math.abs(jt_1 - jt);

            if (idx % 5 == 0) {
                LOGGER.info("Objective: " + jt_1 + "  delta:" + delta);
            }

            // Update Matrix;
            if (delta < REL_ERROR) {
                break;
            }
            jt = jt_1;
        }

        RealMatrix[] outputMatrix = new RealMatrix[2];
        outputMatrix[0] = gammak.transpose().multiply(gammak);
        outputMatrix[1] = nuk.transpose().multiply(nuk);

        // Project
        return outputMatrix;
    }

    /**
     *
     * @param kValue
     * @return
     */
    private void initializeNeighbors(int kValue) {

        Map<String, Map<Integer, RealMatrix>> classMembers = LabelHandling
                .sortIntoMatrixMaps(mapOfPatterns, mapOfClasses);

        RealMatrix tmpMatrix = mapOfPatterns.values().iterator()
                .next();

        RealMatrix uk = MatrixUtils.createRealIdentityMatrix(tmpMatrix.getColumnDimension());

        RealMatrix vk = MatrixUtils.createRealIdentityMatrix(tmpMatrix.getRowDimension());

        MetricDistance_MV metricDistance = new MetricDistance_MV(uk, vk);

        classMemberNear = new HashMap<>();
        // ============= Neighbors =====================
        for (String labelSimilar : classMembers.keySet()) {

            Map<Integer, RealMatrix> mapOfSimilar
                    = classMembers.get(labelSimilar);

            // ==========================================
            Map<Integer, Double> setOfDistances = new HashMap<>();

            for (Integer idx : mapOfSimilar.keySet()) {

                RealMatrix x_i = mapOfSimilar.get(idx);

                // pair wise distances
                mapOfSimilar.keySet().stream()
                        .filter((jdx) -> !(idx.compareTo(jdx) == 0))
                        .forEachOrdered((jdx) -> {
                            setOfDistances.put(jdx,
                                    metricDistance.matrixDistance(x_i, mapOfSimilar.get(jdx)));
                        });

                Map<Integer, Double> sortedDistances
                        = SortingOperations.sortByAcendingValue(setOfDistances);

                List<Integer> setOfNeighbors = new ArrayList<>();
                int counter = 0;
                for (Integer jdx : sortedDistances.keySet()) {
                    setOfNeighbors.add(jdx);
                    counter++;
                    if (counter == kValue) {
                        break;
                    }
                }
                classMemberNear.put(idx, setOfNeighbors);
            }
        }
    }

    /**
     *
     * @param kValue
     * @return
     */
    private void initializeNeighbors_ECVA(int kValue) {

        Map<String, List<Integer>> classMembers = LabelHandling
                .sortIntoMaps(mapOfClasses);

        Map<Integer, RealVector> setOfPatterns_Training = new HashMap<>();

        for (Integer idx : mapOfClasses.keySet()) {
            setOfPatterns_Training.put(idx, new ArrayRealVector(
                    MatrixOperations.unpackMatrix(mapOfPatterns.get(idx))));
        }

        List<Integer> keysTraining = new ArrayList<>(mapOfClasses.keySet());

        Map<Integer, RealVector> setOfData = setOfPatterns_Training;

        ECVA ecva = new ECVA(keysTraining.stream()
                .filter(setOfData::containsKey)
                .collect(Collectors.toMap(Function.identity(), setOfData::get)),
                mapOfClasses);

        CanonicalVariates cva = ecva.execute();

        classMemberNear = new HashMap<>();
        // ============= Neighbors =====================
        for (String labelSimilar : classMembers.keySet()) {

            List<Integer> mapOfSimilar = classMembers.get(labelSimilar);

            // ==========================================
            for (Integer idx : mapOfSimilar) {
                Map<Integer, Double> setOfDistances = new HashMap<>();

                mapOfSimilar.stream().filter((jdx) -> !(idx == jdx)).forEachOrdered((jdx) -> {
                    setOfDistances.put(jdx, cva.getCanonicalVariates().get(idx).getDistance(
                            cva.getCanonicalVariates().get(jdx)));
                });

                Map<Integer, Double> sortedDistances
                        = SortingOperations.sortByAcendingValue(setOfDistances);

                List<Integer> setOfNeighbors = new ArrayList<>();
                int counter = 0;
                for (Integer jdx : sortedDistances.keySet()) {
                    setOfNeighbors.add(jdx);
                    counter++;
                    if (counter == kValue) {
                        break;
                    }
                }
                classMemberNear.put(idx, setOfNeighbors);
            }
        }
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
