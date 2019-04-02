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
package fit.astro.vsa.utilities.ml.knn;

import fit.astro.vsa.common.bindings.math.ml.metric.MultiViewMetric;
import fit.astro.vsa.common.datahandling.LabelHandling;
import fit.astro.vsa.common.utilities.math.support.SortingOperations;
import fit.astro.vsa.utilities.ml.MultiViewMetricDistance;
import fit.astro.vsa.utilities.ml.ecva.CanonicalVariates;
import fit.astro.vsa.utilities.ml.ecva.ECVA;
import fit.astro.vsa.utilities.ml.metriclearning.l3ml.L3MLVariable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author kjohnston
 */
public class GenerateNeighbors_Multi {

    // Input
    private final Map<Integer, Map<String, RealVector>> mapOfPatterns;
    private final Map<Integer, String> mapOfClasses;

    // Internal
    private final Set<String> features;

    /**
     *
     * @param mapOfPatterns
     * @param mapOfClasses
     */
    public GenerateNeighbors_Multi(
            Map<Integer, Map<String, RealVector>> mapOfPatterns,
            Map<Integer, String> mapOfClasses) {
        this.mapOfPatterns = mapOfPatterns;
        this.mapOfClasses = mapOfClasses;
        int idx = mapOfPatterns.keySet().iterator().next();
        this.features = mapOfPatterns.get(idx).keySet();
    }

    /**
     *
     * @param kValue
     * @return
     */
    public Map<Integer, List<Integer>> kNN(int kValue) {

        Map<String, RealVector> startingSet = mapOfPatterns.values().iterator()
                .next();

        Map<String, L3MLVariable> l3mlVariables = new HashMap<>(features.size());

        features.forEach((idx) -> {
            l3mlVariables.put(idx, new L3MLVariable(
                    MatrixUtils.createRealIdentityMatrix(
                            startingSet.get(idx).getDimension()),
                    1.0 / (double) startingSet.size(), 0.0, 0.0));
        });

        Map<Integer, List<Integer>> classMemberNear = new HashMap<>();

        Map<String, List<Integer>> classMembers = LabelHandling
                .sortIntoMaps(mapOfClasses);

        Map<String, MultiViewMetric> inputTmp = new HashMap<>(features.size());
        for (String view : features) {
            L3MLVariable var = l3mlVariables.get(view);

            inputTmp.put(view, new MultiViewMetric(
                    var.getLk().transpose().multiply(var.getLk()),
                    var.getWeight()));
        }

        MultiViewMetricDistance matrix_MetricDistance
                = new MultiViewMetricDistance(inputTmp);

        // ============= Neighbors =====================
        for (String labelSimilar : classMembers.keySet()) {

            List<Integer> mapOfSimilar = classMembers.get(labelSimilar);

            // ==========================================
            for (Integer idx : mapOfSimilar) {

                Map<String, RealVector> x_i = mapOfPatterns.get(idx);

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
        return classMemberNear;
    }

    /**
     *
     * @param kValue
     * @return
     */
    public Map<Integer, List<Integer>> ecva_MV(int kValue) {

        Map<Integer, List<Integer>> classMemberNear = new HashMap<>();

        Map<String, List<Integer>> classMembers = LabelHandling
                .sortIntoMaps(mapOfClasses);

        Map<String, Map<Integer, RealVector>> setOfPatterns_Training = new HashMap<>(features.size());

        for (String view : features) {
            setOfPatterns_Training.put(view, new HashMap<>(mapOfClasses.keySet().size()));
            mapOfClasses.keySet().forEach((idx) -> {
                RealVector tmp = mapOfPatterns.get(idx).get(view);
                setOfPatterns_Training.get(view).put(idx, tmp);
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

        return classMemberNear;
    }

}
