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
package fit.astro.vsa.utilities.ml.metriclearning.l3ml_mv;

import fit.astro.vsa.common.utilities.math.support.SortingOperations;
import fit.astro.vsa.common.bindings.ml.metric.MultiViewMetric;
import fit.astro.vsa.common.bindings.ml.metric.MultiViewMetric_MV;
import fit.astro.vsa.utilities.ml.MultiViewMetricDistance_MV;
import fit.astro.vsa.common.datahandling.LabelHandling;
import fit.astro.vsa.common.utilities.math.linearalgebra.MatrixOperations;
import fit.astro.vsa.utilities.ml.MultiViewMetricDistance;
import fit.astro.vsa.utilities.ml.ecva.CanonicalVariates;
import fit.astro.vsa.utilities.ml.ecva.ECVA;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
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
 * Hu, J., Lu, J., Yuan, J., & Tan, Y. P. (2014, November). Large margin
 * multi-metric learning for face and kinship verification in the wild. In Asian
 * Conference on Computer Vision (pp. 252-267). Springer, Cham.
 *
 * @author Kyle Johnston
 */
public class LargeMarginMultiMetricLearning_MV {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(LargeMarginMultiMetricLearning_MV.class);

    private final Map<Integer, Map<String, RealMatrix>> mapOfPatterns;
    private final Map<Integer, String> mapOfClasses;

    private Map<Integer, List<Integer>> classMemberNear;

    private final Set<String> features;

    private int MAX_ITER = Integer.MAX_VALUE;

    private double REL_ERROR = 1e-1;

    private double BETA_U = 5e-11;
    private double BETA_V = 5e-11;

    //============================================================ 
    private double LAMBDA = 0.5;
    private double GAMMA = 0.5;
    private double MU = 0.5;

    /**
     *
     * @param mapOfPatterns
     * @param mapOfClasses
     */
    public LargeMarginMultiMetricLearning_MV(
            Map<Integer, Map<String, RealMatrix>> mapOfPatterns,
            Map<Integer, String> mapOfClasses) {
        this.mapOfClasses = mapOfClasses;
        this.mapOfPatterns = mapOfPatterns;
        int idx = mapOfPatterns.keySet().iterator().next();
        this.features = mapOfPatterns.get(idx).keySet();
    }

    /**
     *
     * @param knn
     * @return
     */
    public Map<String, MultiViewMetric_MV> execute(int knn) {

        Map<String, RealMatrix> startingSet = mapOfPatterns.values().iterator()
                .next();

        int K = startingSet.size();

        //======================================================
        // Initialize L Matrix (Covariance Matrix for) to the identity matrix
        Map<String, L3MLVariable_MV> l3mlVariables = new HashMap<>(features.size());

        Map<String, RealMatrix> gradU_k = new HashMap<>(features.size());
        Map<String, RealMatrix> gradV_k = new HashMap<>(features.size());

        Map<String, RealMatrix> n_k = new HashMap<>(features.size());
        Map<String, RealMatrix> g_k = new HashMap<>(features.size());

        for (String idx : features) {
            RealMatrix gammak = MatrixUtils.createRealIdentityMatrix(
                    startingSet.get(idx).getColumnDimension()).scalarMultiply(1e-1);
            
            RealMatrix nuk = MatrixUtils.createRealIdentityMatrix(
                    startingSet.get(idx).getRowDimension()).scalarMultiply(1e-1);
            
            g_k.put(idx, gammak);
            n_k.put(idx, nuk);

            gradU_k.put(idx, gammak);
            gradV_k.put(idx, nuk);

            l3mlVariables.put(idx, new L3MLVariable_MV(gammak, nuk, 1.0 / (double) K));
        }

        initializeNeighbors_Mean(knn);

        //======================================================
        L3ML_MV_MetricLearningObjective learningObjective
                = new L3ML_MV_MetricLearningObjective(classMemberNear,
                        mapOfPatterns, mapOfClasses);
        learningObjective.setLAMBDA_REG(LAMBDA);
        learningObjective.setGAMMA_PP(GAMMA);
        learningObjective.setMU_CR(MU);

        L3ML_MV_MetricLearningGradientGenerator learningGradientGenerator
                = new L3ML_MV_MetricLearningGradientGenerator(classMemberNear,
                        mapOfPatterns, mapOfClasses);
        learningGradientGenerator.setLAMBDA(LAMBDA);
        learningGradientGenerator.setGAMMA(GAMMA);
        learningGradientGenerator.setMU(MU);

        double jt = 0;
        LOGGER.info("Objective Function is to be minimized");
        LOGGER.info("Every Other Step Logged");

        for (int idx = 0; idx < MAX_ITER; idx++) {

            double step_u;
            double step_v;

            // Step 2 Update Lk
            for (String kdx : features) {

                RealMatrix[] gradiantOfJwrtLMatrix = learningGradientGenerator
                        .generateLk(kdx, l3mlVariables);

                // ======
                if (idx == 0) {
                    step_u = BETA_U;
                    step_v = BETA_V;
                } else {
                    step_u = generateBeta_BB(gradiantOfJwrtLMatrix[0],
                            gradU_k.get(kdx), l3mlVariables.get(kdx).getGammak(), g_k.get(kdx));

                    step_v = generateBeta_BB(gradiantOfJwrtLMatrix[1],
                            gradV_k.get(kdx), l3mlVariables.get(kdx).getNuk(), n_k.get(kdx));
                }

                LOGGER.info("step_u: " + step_u + "  step_v:" + step_v);
                
                // ======
                RealMatrix tmpG = l3mlVariables.get(kdx).getGammak()
                        .subtract(gradiantOfJwrtLMatrix[0].scalarMultiply(step_u));
                g_k.put(kdx, l3mlVariables.get(kdx).getGammak());
                gradU_k.put(kdx, gradiantOfJwrtLMatrix[0]);
                l3mlVariables.get(kdx).setGammak(tmpG);

                // ======
                RealMatrix tmpN = l3mlVariables.get(kdx).getNuk()
                        .subtract(gradiantOfJwrtLMatrix[1].scalarMultiply(step_v));
                n_k.put(kdx, l3mlVariables.get(kdx).getNuk());
                gradV_k.put(kdx, gradiantOfJwrtLMatrix[1]);
                l3mlVariables.get(kdx).setNuk(tmpN);

            }

            // ======
            // Step 3 Update wk
//            Map<String, Double> ikMap = learningGradientGenerator.updateWeight(l3mlVariables);

            // estimate delta opt change
            double jt_1 = 0;
            for (String kdx : features) {
                double tmp = learningObjective.valueJK(kdx, l3mlVariables);
                LOGGER.info("Objective: " + tmp + "  kdx: " + kdx);
                jt_1 += tmp;
            }

            double delta = Math.abs(jt_1 - jt);

            if (idx % 1 == 0) {
                LOGGER.info("Objective: " + jt_1 + "  delta:" + delta);
            }

            if (delta < REL_ERROR) {
                break;
            }

            jt = jt_1;
        }

        Map<String, MultiViewMetric_MV> outputVar = new HashMap<>(features.size());

        for (String feature : features) {
            RealMatrix uk = l3mlVariables.get(feature).getUk();
            RealMatrix vk = l3mlVariables.get(feature).getVk();

            outputVar.put(feature, new MultiViewMetric_MV(uk, vk,
                    l3mlVariables.get(feature).getWeight()));
        }

        return outputVar;
    }

    /**
     *
     * @param kValue
     * @return
     */
    private void initializeNeighbors(Map<String, L3MLVariable_MV> l3mlVariables, int kValue) {

        Map<String, List<Integer>> classMembers = LabelHandling
                .sortIntoMaps(mapOfClasses);

        Map<String, MultiViewMetric_MV> inputTmp = new HashMap<>(features.size());
        for (String view : features) {
            L3MLVariable_MV var = l3mlVariables.get(view);

            inputTmp.put(view, new MultiViewMetric_MV(
                    var.getGammak().transpose().multiply(var.getGammak()),
                    var.getNuk().transpose().multiply(var.getNuk()),
                    var.getWeight()));
        }

        MultiViewMetricDistance_MV matrix_MetricDistance
                = new MultiViewMetricDistance_MV(inputTmp);

        classMemberNear = new HashMap<>();
        // ============= Neighbors =====================
        for (String labelSimilar : classMembers.keySet()) {

            List<Integer> mapOfSimilar = classMembers.get(labelSimilar);

            // ==========================================
            for (Integer idx : mapOfSimilar) {

                Map<String, RealMatrix> x_i = mapOfPatterns.get(idx);

                Map<Integer, Double> setOfDistances = new HashMap<>();

                // pair wise distances
                mapOfSimilar.stream().filter((jdx) -> !(idx.compareTo(jdx) == 0))
                        .forEachOrdered((jdx) -> {
                            setOfDistances.put(jdx, matrix_MetricDistance
                                    .multiviewDistance(x_i, mapOfPatterns.get(jdx)));
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
    private void initializeNeighbors_Mean(int kValue) {

        Map<String, List<Integer>> classMembers = LabelHandling
                .sortIntoMaps(mapOfClasses);

        Map<String, MultiViewMetric_MV> inputTmp = generateIdentity();

        classMemberNear = new HashMap<>();

        Map<String, Map<String, RealMatrix>> meanPerClass = new HashMap<>(classMembers.keySet().size());
        // ============= Neighbors =====================
        for (String labelSimilar : classMembers.keySet()) {

            List<Integer> mapOfSimilar = classMembers.get(labelSimilar);

            // ==========================================
            Map<String, RealMatrix> meanVariate = new HashMap<>();
            inputTmp.keySet().forEach((view) -> {
                meanVariate.put(view, new Array2DRowRealMatrix(inputTmp.get(view).getVk().getRowDimension(),
                        inputTmp.get(view).getUk().getColumnDimension()));
            });

            // ==========================================
            mapOfSimilar.stream().map((idx) -> mapOfPatterns.get(idx)).forEachOrdered((x_i) -> {
                x_i.keySet().forEach((view) -> {
                    meanVariate.put(view, meanVariate.get(view).add(x_i.get(view)));
                });
            });

            // ==========================================
            meanVariate.keySet().forEach((view) -> {
                meanVariate.put(view, meanVariate.get(view).scalarMultiply(1.0 / (double) mapOfSimilar.size()));
            });

            // ==========================================
            meanPerClass.put(labelSimilar, meanVariate);
        }

        List<String> classList = new ArrayList<>(meanPerClass.keySet());

        Map<Integer, Map<String, RealVector>> transformedData = new HashMap<>();
        for (Integer idx : mapOfPatterns.keySet()) {

            Map<String, RealMatrix> patterns = mapOfPatterns.get(idx);

            Map<String, RealVector> transformedPattern = new HashMap<>();
            for (String view : patterns.keySet()) {
                RealVector distancePattern = new ArrayRealVector(meanPerClass.keySet().size());

                for (int jdx = 0; jdx < classList.size(); jdx++) {
                    RealMatrix mean = meanPerClass.get(classList.get(jdx)).get(view);
                    RealMatrix pattern = patterns.get(view);

                    double distance = (mean.subtract(pattern)).getFrobeniusNorm();
                    distancePattern.setEntry(jdx, distance);
                }

                transformedPattern.put(view, distancePattern);
            }

            transformedData.put(idx, transformedPattern);

        }

        // Create Identity
        Map<String, MultiViewMetric> multiMatrix = new HashMap<>();
        for (String view : inputTmp.keySet()) {

            multiMatrix.put(view, new MultiViewMetric(
                    MatrixUtils.createRealIdentityMatrix(classList.size()), 1.0 / (double) inputTmp.keySet().size()));
        }

        MultiViewMetricDistance metricDistance
                = new MultiViewMetricDistance(multiMatrix);

        for (Integer idx : mapOfPatterns.keySet()) {

            Map<Integer, Double> setOfDistances = new HashMap<>();

            Map<String, RealVector> xi = transformedData.get(idx);
            mapOfPatterns.keySet().stream().filter((jdx) -> (!Objects.equals(idx, jdx))).forEachOrdered((jdx) -> {
                setOfDistances.put(jdx, metricDistance.multiviewDistance(xi, transformedData.get(jdx)));
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

    /**
     *
     * @param kValue
     * @return
     */
    private void initializeNeighbors(int kValue) {

        Map<String, List<Integer>> classMembers = LabelHandling
                .sortIntoMaps(mapOfClasses);

        Map<String, Map<Integer, RealVector>> setOfPatterns_Training = new HashMap<>(features.size());

        for (String view : features) {
            setOfPatterns_Training.put(view, new HashMap<>(mapOfClasses.keySet().size()));
            mapOfClasses.keySet().forEach((idx) -> {
                RealMatrix tmp = mapOfPatterns.get(idx).get(view);
                setOfPatterns_Training
                        .get(view).put(idx, new ArrayRealVector(
                        MatrixOperations.unpackMatrix(tmp)));

            });

        }

        List<Integer> keysTraining = new ArrayList<>(mapOfClasses.keySet());
        Map<String, CanonicalVariates> cva = new HashMap<>(features.size());

        for (String view : features) {

            Map<Integer, RealVector> setOfData = setOfPatterns_Training.get(view);

            ECVA ecva = new ECVA(keysTraining.stream()
                    .filter(setOfData::containsKey)
                    .collect(Collectors.toMap(Function.identity(), setOfData::get)),
                    mapOfClasses);

            cva.put(view, ecva.execute());

        }

        classMemberNear = new HashMap<>();
        // ============= Neighbors =====================

        for (String labelSimilar : classMembers.keySet()) {

            List<Integer> mapOfSimilar = classMembers.get(labelSimilar);

            // ==========================================
            for (Integer idx : mapOfSimilar) {
                Map<Integer, Double> setOfDistances = new HashMap<>();

                // pair wise distances
                mapOfSimilar.stream().filter((jdx) -> !(idx == jdx)).forEachOrdered((jdx) -> {
                    double distance = 0;
                    distance = features.stream().map((views) -> cva.get(views).getCanonicalVariates().get(idx).getDistance(
                            cva.get(views).getCanonicalVariates().get(jdx)))
                            .reduce(distance, (accumulator, _item) -> accumulator + _item);

                    setOfDistances.put(jdx, distance);

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
     * Generate the identity matrices for the MV case.
     *
     * @return
     */
    public Map<String, MultiViewMetric_MV> generateIdentity() {
        Map<String, RealMatrix> startingSet = mapOfPatterns.values().iterator()
                .next();

        int K = startingSet.size();

        //======================================================
        // Initialize L Matrix (Covariance Matrix for) to the identity matrix
        Map<String, L3MLVariable_MV> l3mlVariables = new HashMap<>(features.size());

        Map<String, RealMatrix> gradU_k = new HashMap<>(features.size());
        Map<String, RealMatrix> gradV_k = new HashMap<>(features.size());

        Map<String, RealMatrix> n_k = new HashMap<>(features.size());
        Map<String, RealMatrix> g_k = new HashMap<>(features.size());

        for (String idx : features) {
            RealMatrix gammak = MatrixUtils.createRealIdentityMatrix(
                    startingSet.get(idx).getColumnDimension());

            RealMatrix nuk = MatrixUtils.createRealIdentityMatrix(
                    startingSet.get(idx).getRowDimension());

            g_k.put(idx, gammak);
            n_k.put(idx, nuk);

            gradU_k.put(idx, gammak);
            gradV_k.put(idx, nuk);

            l3mlVariables.put(idx, new L3MLVariable_MV(gammak, nuk, 1.0 / (double) K));

        }

        Map<String, MultiViewMetric_MV> outputVar = new HashMap<>(features.size());

        for (String feature : features) {
            RealMatrix uk = l3mlVariables.get(feature).getGammak().transpose().multiply(
                    l3mlVariables.get(feature).getGammak());

            RealMatrix vk = l3mlVariables.get(feature).getNuk().transpose().multiply(
                    l3mlVariables.get(feature).getNuk());

            outputVar.put(feature, new MultiViewMetric_MV(uk, vk,
                    l3mlVariables.get(feature).getWeight()));

        }

        return outputVar;

    }

    public void setBETA_U(double BETA_U) {
        this.BETA_U = BETA_U;

    }

    public void setBETA_V(double BETA_V) {
        this.BETA_V = BETA_V;

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
     * @param GAMMA
     */
    public void setGAMMA(double GAMMA) {
        this.GAMMA = GAMMA;

    }

    /**
     *
     * @param MU
     */
    public void setMU(double MU) {
        this.MU = MU;

    }

    /**
     *
     * @param REL_ERROR
     */
    public void setREL_ERROR(double REL_ERROR) {
        this.REL_ERROR = REL_ERROR;
    }

}
