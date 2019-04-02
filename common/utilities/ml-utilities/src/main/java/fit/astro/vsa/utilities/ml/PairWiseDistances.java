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
package fit.astro.vsa.utilities.ml;

import fit.astro.vsa.common.bindings.math.vector.VectorDistanceType;
import fit.astro.vsa.utilities.ml.distance.PearsonDistance;
import java.util.List;
import java.util.stream.IntStream;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.ml.distance.CanberraDistance;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.ml.distance.ManhattanDistance;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class PairWiseDistances {

    private static final Logger LOGGER = LoggerFactory.getLogger(PairWiseDistances.class);

    // ======== Input
    private final List<RealVector> inputObservations;
    
    // ======== Internal
    private final int[] nBins;

    /**
     *
     * @param inputObservations
     */
    public PairWiseDistances(List<RealVector> inputObservations) {
        this.inputObservations = inputObservations;

        // Mutual information distance.
        this.nBins = new int[29];
        int counter = 0;
        for (int i = 8; i <= 64; i = i + 2) {
            nBins[counter] = i;
            counter++;
        }
    }

    /**
     *
     * @param distanceMetricType
     * <p>
     * @return
     */
    public RealMatrix generateDistances(
            VectorDistanceType distanceMetricType) {

        RealMatrix distanceMatrix = MatrixUtils.createRealMatrix(
                inputObservations.size(), inputObservations.size());

        /**
         * Upper Triangular Distance Matrix
         */
        for (int i = 0; i < inputObservations.size(); i++) {
            RealVector prime = inputObservations.get(i);
            for (int j = i + 1; j < inputObservations.size(); j++) {

                distanceMatrix.setEntry(i, j,
                        estimateDistance(prime,
                                inputObservations.get(j),
                                distanceMetricType));
            }

        }
        //
        return distanceMatrix.add(distanceMatrix.transpose());
    }

    /**
     *
     * @param distanceMetricType
     * <p>
     * @return
     */
    public RealMatrix generateDistancesParallel(
            VectorDistanceType distanceMetricType) {

        RealMatrix distanceMatrix = MatrixUtils.createRealMatrix(
                inputObservations.size(), inputObservations.size());

        /**
         * Upper Triangular Distance Matrix
         */
        IntStream.range(0, inputObservations.size()).parallel().forEach(idx -> {
            RealVector prime = inputObservations.get(idx);

            for (int j = idx + 1; j < inputObservations.size(); j++) {
                distanceMatrix.setEntry(idx, j,
                        estimateDistance(prime, inputObservations.get(j),
                                distanceMetricType));
            }

        });

        return distanceMatrix.add(distanceMatrix.transpose());
    }

    /**
     *
     * @param prime
     * @param other
     * @param distanceMetricType
     * @return
     */
    private double estimateDistance(RealVector prime, RealVector other,
            VectorDistanceType distanceMetricType) {
        double distance;
        DistanceMeasure distMeasure;
        
        switch (distanceMetricType) {
            
            case CANBERRA_DISTANCE:
                distMeasure = new CanberraDistance();
                break;
            case CITY_BLOCK:
                distMeasure = new ManhattanDistance();
                break;
            case EUCLIDEAN_DISTANCE:
                distMeasure = new EuclideanDistance();
                break;
            case PEARSON_DISTANCE:
                distMeasure = new PearsonDistance();
                break;
            default:
                LOGGER.debug("Distance Type: "
                        + distanceMetricType.getMethodLabel()
                        + " not for use in PairWise Distances");
                throw new ArithmeticException("Distance Type: "
                        + distanceMetricType.getMethodLabel()
                        + " not for use in PairWise Distances");
        }

        distance =  distMeasure.compute(prime.toArray(), other.toArray());
        
        return distance;
    }

}
