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

import fit.astro.vsa.common.datahandling.LabelHandling;
import fit.astro.vsa.common.utilities.math.support.SortingOperations;
import fit.astro.vsa.utilities.ml.MetricDistance;
import fit.astro.vsa.utilities.ml.ecva.CanonicalVariates;
import fit.astro.vsa.utilities.ml.ecva.ECVA;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author kjohnston
 */
public class GenerateNeighbors {

    // Input
    private final Map<Integer, RealVector> mapOfPatterns;
    private final Map<Integer, String> mapOfClasses;

    /**
     *
     * @param mapOfPatterns
     * @param mapOfClasses
     */
    public GenerateNeighbors(
            Map<Integer, RealVector> mapOfPatterns,
            Map<Integer, String> mapOfClasses) {
        this.mapOfPatterns = mapOfPatterns;
        this.mapOfClasses = mapOfClasses;
    }

    /**
     *
     * @param kValue
     * @return
     */
    public Map<Integer, List<Integer>> kNN(int kValue) {

        RealVector startingSet = mapOfPatterns.values().iterator()
                .next();

        Map<Integer, List<Integer>> classMemberNear = new HashMap<>();

        Map<String, List<Integer>> classMembers = LabelHandling
                .sortIntoMaps(mapOfClasses);

        MetricDistance metricDistance
                = new MetricDistance(MatrixUtils.createRealIdentityMatrix(startingSet.getDimension()));

        // ============= Neighbors =====================
        for (String labelSimilar : classMembers.keySet()) {

            List<Integer> mapOfSimilar = classMembers.get(labelSimilar);

            // ==========================================
            for (Integer idx : mapOfSimilar) {

                RealVector x_i = mapOfPatterns.get(idx);

                Map<Integer, Double> setOfDistances = new HashMap<>();

                // pair wise distances
                mapOfSimilar.stream().filter((jdx) -> !(idx.compareTo(jdx) == 0))
                        .forEachOrdered((jdx) -> {
                            setOfDistances.put(jdx, metricDistance
                                    .distance(x_i, mapOfPatterns.get(jdx)));
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

        ECVA ecva = new ECVA(mapOfPatterns, mapOfClasses);

        CanonicalVariates cva = ecva.execute();

        // ============= Neighbors =====================
        for (String labelSimilar : classMembers.keySet()) {

            List<Integer> mapOfSimilar = classMembers.get(labelSimilar);

            // ==========================================
            for (Integer idx : mapOfSimilar) {
                Map<Integer, Double> setOfDistances = new HashMap<>();

                // pair wise distances
                mapOfSimilar.stream().filter((jdx) -> !(idx == jdx)).forEachOrdered((jdx) -> {
                    double distance = cva.getCanonicalVariates().get(idx).getDistance(
                            cva.getCanonicalVariates().get(jdx));

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
