/*
 * Copyright (C) 2018 Kyle Johnston <kyjohnst2000@my.fit.edu>
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

import fit.astro.vsa.common.utilities.math.linearalgebra.MatrixOperations;
import fit.astro.vsa.common.bindings.math.matrix.UnivariateFunctionMapper;
import java.util.HashMap;
import java.util.Map;
import org.apache.commons.math3.analysis.function.Power;
import org.apache.commons.math3.analysis.function.Sqrt;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.Pair;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class NormalizeData {

    /**
     * Generate the Normalization Vectors
     *
     * @param mapOfPatterns Input Patterns
     * @return Pairing of Mean and Std Vector For Normalization
     */
    public static Pair<Map<String, RealMatrix>, Map<String, RealMatrix>> normalizeMultiViewMatrixVariate(
            Map<Integer, Map<String, RealMatrix>> mapOfPatterns) {

        Map<String, RealMatrix> meanPattern = new HashMap<>();
        Map<String, RealMatrix> stdPattern = new HashMap<>();

        Integer next = mapOfPatterns.keySet().iterator().next();
        Map<String, RealMatrix> tmpPattern = mapOfPatterns.get(next);

        for (String feature : tmpPattern.keySet()) {

            RealMatrix sumPatterns = new Array2DRowRealMatrix(
                    tmpPattern.get(feature).getRowDimension(),
                    tmpPattern.get(feature).getColumnDimension());

            RealMatrix sumPatternsSq = new Array2DRowRealMatrix(
                    tmpPattern.get(feature).getRowDimension(),
                    tmpPattern.get(feature).getColumnDimension());

            double n = (double) mapOfPatterns.size();

            for (Integer idx : mapOfPatterns.keySet()) {
                RealMatrix x = mapOfPatterns.get(idx).get(feature);
                sumPatterns = sumPatterns.add(x);
                sumPatternsSq = sumPatternsSq.add(MatrixOperations.hadamardProduct(x, x));
            }

            RealMatrix mean = sumPatterns.scalarMultiply(1.0 / n);

            RealMatrix squOfSum = sumPatterns.copy();
            squOfSum.walkInOptimizedOrder(new UnivariateFunctionMapper(new Power(2.0)));

            RealMatrix temp = (sumPatternsSq.scalarMultiply(n)).subtract(squOfSum);

            RealMatrix variance = temp.scalarMultiply(1.0 / (n * (n - 1.0)));

            meanPattern.put(feature, mean);

            RealMatrix std = variance.copy();
            std.walkInOptimizedOrder(new UnivariateFunctionMapper(new Sqrt()));

            stdPattern.put(feature, std);
        }

        return new Pair<>(meanPattern, stdPattern);
    }

    /**
     * Apply the Normalization Vectors
     *
     * @param mapOfPatterns The input patterns
     * @param transformVectors NOrmalization Vectors (Pair of Mean and Std)
     * @return The transformed patterns
     */
    public static Map<Integer, Map<String, RealMatrix>> applyNormalizeMatrixVariate(
            Map<Integer, Map<String, RealMatrix>> mapOfPatterns,
            Pair<Map<String, RealMatrix>, Map<String, RealMatrix>> transformVectors) {

        // Normalized Data Set to be Operated On
        Map<Integer, Map<String, RealMatrix>> mapOfScaledPatterns = new HashMap<>(mapOfPatterns.size());

        Integer next = mapOfPatterns.keySet().iterator().next();
        Map<String, RealMatrix> tmpPattern = mapOfPatterns.get(next);

        for (Integer idx : mapOfPatterns.keySet()) {

            mapOfScaledPatterns.put(idx, new HashMap<>(tmpPattern.keySet().size()));

            for (String feature : tmpPattern.keySet()) {
                // Normalize Data
                RealMatrix a = mapOfPatterns.get(idx).get(feature)
                        .subtract(transformVectors.getFirst().get(feature));

                RealMatrix c = MatrixOperations.hadamardDivision(a,
                        transformVectors.getSecond().get(feature));

                mapOfScaledPatterns.get(idx).put(feature, c);

            }
        }

        return mapOfScaledPatterns;
    }

    /**
     * Apply the Normalization Vectors
     *
     * @param mapOfPatterns The input patterns
     * @param transformVectors NOrmalization Vectors (Pair of Mean and Std)
     * @return The transformed patterns
     */
    public static Map<Integer, RealMatrix> applyNormMatrix(
            Map<Integer, RealMatrix> mapOfPatterns,
            Pair<RealMatrix, RealMatrix> transformVectors) {

        // Normalized Data Set to be Operated On
        Map<Integer, RealMatrix> mapOfScaledPatterns = new HashMap<>(mapOfPatterns.size());

        for (Integer idx : mapOfPatterns.keySet()) {

            // Normalize Data
            RealMatrix a = mapOfPatterns.get(idx)
                    .subtract(transformVectors.getFirst());

            RealMatrix c = MatrixOperations.hadamardDivision(a,
                    transformVectors.getSecond());

            mapOfScaledPatterns.put(idx, c);

        }

        return mapOfScaledPatterns;
    }

    /**
     * Generate the Normalization Vectors
     *
     * @param mapOfPatterns Input Patterns
     * @return Pairing of Mean and Std Vector For Normalization
     */
    public static Pair<Map<String, RealVector>, Map<String, RealVector>> normalizeMultiViewVectorVariate(
            Map<Integer, Map<String, RealVector>> mapOfPatterns) {

        Map<String, RealVector> meanPattern = new HashMap<>();
        Map<String, RealVector> stdPattern = new HashMap<>();

        Integer next = mapOfPatterns.keySet().iterator().next();
        Map<String, RealVector> tmpPattern = mapOfPatterns.get(next);

        for (String feature : tmpPattern.keySet()) {

            RealVector sumPatterns = new ArrayRealVector(tmpPattern.get(feature).getDimension());
            RealVector sumPatternsSq = new ArrayRealVector(tmpPattern.get(feature).getDimension());

            for (Integer idx : mapOfPatterns.keySet()) {
                RealVector x = mapOfPatterns.get(idx).get(feature);
                sumPatterns = sumPatterns.add(x);
                sumPatternsSq = sumPatternsSq.add(x.ebeMultiply(x));
            }

            RealVector mean = sumPatterns.mapDivide((double) mapOfPatterns.size());

            RealVector variance = sumPatternsSq
                    .subtract(sumPatterns.map(new Power(2.0))
                            .mapDivide((double) mapOfPatterns.size()))
                    .mapDivide((double) mapOfPatterns.size() - 1.0);

            meanPattern.put(feature, mean);
            stdPattern.put(feature, variance.map(new Sqrt()));
        }

        return new Pair<>(meanPattern, stdPattern);
    }

    /**
     * Apply the Normalization Vectors
     *
     * @param mapOfPatterns The input patterns
     * @param transformVectors NOrmalization Vectors (Pair of Mean and Std)
     * @return The transformed patterns
     */
    public static Map<Integer, Map<String, RealVector>> applyNormalizeVectorVariate(
            Map<Integer, Map<String, RealVector>> mapOfPatterns,
            Pair<Map<String, RealVector>, Map<String, RealVector>> transformVectors) {

        // Normalized Data Set to be Operated On
        Map<Integer, Map<String, RealVector>> mapOfScaledPatterns = new HashMap<>(mapOfPatterns.size());

        Integer next = mapOfPatterns.keySet().iterator().next();
        Map<String, RealVector> tmpPattern = mapOfPatterns.get(next);

        for (Integer idx : mapOfPatterns.keySet()) {

            mapOfScaledPatterns.put(idx, new HashMap<>(tmpPattern.keySet().size()));

            for (String feature : tmpPattern.keySet()) {
                // Normalize Data
                mapOfScaledPatterns.get(idx).put(feature, (mapOfPatterns.get(idx).get(feature)
                        .subtract(transformVectors.getFirst().get(feature)))
                        .ebeDivide(transformVectors.getSecond().get(feature)));

                if (mapOfScaledPatterns.get(idx).get(feature).isNaN()) {
                    for (int jdx = 0; jdx < mapOfScaledPatterns.get(idx).get(feature).getDimension(); jdx++) {
                        if (Double.isNaN(mapOfScaledPatterns.get(idx).get(feature).getEntry(jdx))) {
                            mapOfScaledPatterns.get(idx).get(feature).setEntry(jdx, 0);
                        }
                    }
                }
            }
        }

        return mapOfScaledPatterns;
    }

    /**
     * Generate the Normalization Vectors
     *
     * @param mapOfPatterns Input Patterns
     * @return Pairing of Mean and Std Vector For Normalization
     */
    public static Pair<RealVector, RealVector> normalizeVector(
            Map<Integer, RealVector> mapOfPatterns) {

        Integer next = mapOfPatterns.keySet().iterator().next();
        RealVector tmpPattern = mapOfPatterns.get(next);

        RealVector sumPatterns = new ArrayRealVector(tmpPattern.getDimension());
        RealVector sumPatternsSq = new ArrayRealVector(tmpPattern.getDimension());

        for (Integer idx : mapOfPatterns.keySet()) {
            RealVector x = mapOfPatterns.get(idx);
            sumPatterns = sumPatterns.add(x);
            sumPatternsSq = sumPatternsSq.add(x.ebeMultiply(x));
        }

        RealVector mean = sumPatterns.mapDivide((double) mapOfPatterns.size());

        RealVector variance = sumPatternsSq
                .subtract(sumPatterns.map(new Power(2.0))
                        .mapDivide((double) mapOfPatterns.size()))
                .mapDivide((double) mapOfPatterns.size() - 1.0);

        return new Pair<>(mean, variance.map(new Sqrt()));
    }

    /**
     * Generate the Normalization Vectors
     *
     * @param mapOfPatterns Input Patterns
     * @return Pairing of Mean and Std Vector For Normalization
     */
    public static Pair<RealMatrix, RealMatrix> normalizeMatrix(
            Map<Integer, RealMatrix> mapOfPatterns) {

        Integer next = mapOfPatterns.keySet().iterator().next();
        RealMatrix tmpPattern = mapOfPatterns.get(next);

        RealMatrix sumPatterns = new Array2DRowRealMatrix(
                tmpPattern.getRowDimension(),
                tmpPattern.getColumnDimension());

        RealMatrix sumPatternsSq = new Array2DRowRealMatrix(
                tmpPattern.getRowDimension(),
                tmpPattern.getColumnDimension());

        double n = (double) mapOfPatterns.size();

        for (Integer idx : mapOfPatterns.keySet()) {
            RealMatrix x = mapOfPatterns.get(idx);
            sumPatterns = sumPatterns.add(x);
            sumPatternsSq = sumPatternsSq.add(MatrixOperations.hadamardProduct(x, x));
        }

        RealMatrix mean = sumPatterns.scalarMultiply(1.0 / n);

        RealMatrix squOfSum = sumPatterns.copy();
        squOfSum.walkInOptimizedOrder(new UnivariateFunctionMapper(new Power(2.0)));

        RealMatrix temp = (sumPatternsSq.scalarMultiply(n)).subtract(squOfSum);

        RealMatrix variance = temp.scalarMultiply(1.0 / (n * (n - 1.0)));

        RealMatrix meanPattern = mean;

        RealMatrix std = variance.copy();
        std.walkInOptimizedOrder(new UnivariateFunctionMapper(new Sqrt()));

        RealMatrix stdPattern = std;

        return new Pair<>(meanPattern, stdPattern);
    }

    /**
     * Apply the Normalization Vectors
     *
     * @param mapOfPatterns The input patterns
     * @param transformVectors NOrmalization Vectors (Pair of Mean and Std)
     * @return The transformed patterns
     */
    public static Map<Integer, RealVector> applyNormVector(
            Map<Integer, RealVector> mapOfPatterns,
            Pair<RealVector, RealVector> transformVectors) {

        Map<Integer, RealVector> mapOfScaledPatterns
                = new HashMap<>(mapOfPatterns.size());

        for (Integer idx : mapOfPatterns.keySet()) {

            // Normalize Data
            mapOfScaledPatterns.put(idx, mapOfPatterns.get(idx).subtract(transformVectors.getFirst())
                    .ebeDivide(transformVectors.getSecond()));

            if (mapOfScaledPatterns.get(idx).isNaN()) {
                for (int jdx = 0; jdx < mapOfScaledPatterns.get(idx).getDimension(); jdx++) {
                    if (Double.isNaN(mapOfScaledPatterns.get(idx).getEntry(jdx))) {
                        mapOfScaledPatterns.get(idx).setEntry(jdx, 0);
                    }
                }
            }

        }

        return mapOfScaledPatterns;
    }

}
