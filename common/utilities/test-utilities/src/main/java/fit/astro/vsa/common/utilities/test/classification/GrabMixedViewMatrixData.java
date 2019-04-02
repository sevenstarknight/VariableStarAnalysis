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
package fit.astro.vsa.common.utilities.test.classification;

import fit.astro.vsa.common.utilities.math.linearalgebra.VectorOperations;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PrimitiveIterator;
import java.util.Random;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class GrabMixedViewMatrixData {

    private Random random = new Random(42L);

    // =============================================
    //Output
    private final Map<Integer, Map<String, RealMatrix>> setOfPatterns;
    private final Map<Integer, String> setOfClasses;

    public GrabMixedViewMatrixData() throws IOException {

        GrabIrisData grabIrisData = new GrabIrisData();
        GrabStarlightCurves grabStarlightCurves = new GrabStarlightCurves();

        Map<Integer, RealVector> setOfIris = grabIrisData.getSetOfPatterns();
        Map<Integer, RealVector> setOfStarlight = grabStarlightCurves.getSetOfPatterns();

        Map<Integer, String> labelOfIris = grabIrisData.getSetOfClasses();
        Map<Integer, String> labelOfStarlight = grabStarlightCurves.getSetOfClasses();

        Map<String, List<Integer>> starLocations = sortIntoMaps(labelOfStarlight);

        // ===================================================================
        // Both Iris and Starlight have Three Classes
        List<String> uniqueIris = new ArrayList<>(grabIrisData.getUniqueLabels());
        List<String> uniqueStarlight = new ArrayList<>(grabStarlightCurves.getUniqueLabels());

        Map<String, String> pairingOfInputs = new HashMap<>();
        pairingOfInputs.put(uniqueIris.get(0), uniqueStarlight.get(0));
        pairingOfInputs.put(uniqueIris.get(1), uniqueStarlight.get(1));
        pairingOfInputs.put(uniqueIris.get(2), uniqueStarlight.get(2));

        List<Integer> idxs = new ArrayList<>(setOfIris.keySet());
        PrimitiveIterator.OfInt streamIntI = random.ints(0, setOfIris.size()).iterator();

        this.setOfPatterns = new HashMap<>();
        this.setOfClasses = new HashMap<>();

        for (int kdx = 0; kdx < 1000; kdx++) {

            Map<String, RealMatrix> viewSet = new HashMap<>();
            // =================================
            // Get Iris Dataset
            Integer idx = idxs.get(streamIntI.next());
            viewSet.put("Iris", MatrixUtils.createColumnRealMatrix(setOfIris.get(idx).toArray()));
            String irisLabel = labelOfIris.get(idx);

            // ==================================
            // Pair with Random Starlight Curve of Defined Label
            String starLabel = pairingOfInputs.get(irisLabel);
            List<Integer> tmpStarlight = starLocations.get(starLabel);

            PrimitiveIterator.OfInt streamIntJ = random.ints(0, tmpStarlight.size()).iterator();

            Integer jdx = tmpStarlight.get(streamIntJ.next());

            viewSet.put("Starlight",MatrixUtils.createColumnRealMatrix(
                    downSample(setOfStarlight.get(jdx)).toArray()));

            setOfPatterns.put(kdx, viewSet);
            setOfClasses.put(kdx, starLabel);

        }

    }

    /**
     * Quick Box-Car Average of The waveform to DownSample
     *
     * @param input
     * @return
     */
    private static RealVector downSample(RealVector input) {

        int[] spacing = VectorOperations.linearSpace(input.getDimension(), 0, 100);

        RealVector output = new ArrayRealVector(spacing.length - 1);

        for (int idx = 0; idx < spacing.length - 1; idx++) {

            double mean = VectorOperations.mean(input.getSubVector(spacing[idx],
                    spacing[idx + 1] - spacing[idx]));

            output.setEntry(idx, mean);

        }

        return output;

    }

    public static Map<String, List<Integer>> sortIntoMaps(
            Map<Integer, String> mapOfClasses) {
        // ========================================
        // Set up: Split Input Patterns Into Class Groups
        Map<String, List<Integer>> classMembers = new HashMap<>();
        mapOfClasses.keySet().stream().forEach((currentPatternId) -> {
            String classType = mapOfClasses.get(currentPatternId);

            if (classMembers.containsKey(classType)) {
                List<Integer> listOfIntegers
                        = classMembers.get(classType);

                listOfIntegers.add(currentPatternId);
                classMembers.put(classType, listOfIntegers);
            } else {
                List<Integer> listOfIntegers = new ArrayList<>();
                listOfIntegers.add(currentPatternId);
                classMembers.put(classType, listOfIntegers);
            }
        });

        return classMembers;
    }

    public Random getRandom() {
        return random;
    }

    public Map<Integer, Map<String, RealMatrix>> getSetOfPatterns() {
        return setOfPatterns;
    }

    public Map<Integer, String> getSetOfClasses() {
        return setOfClasses;
    }

}
