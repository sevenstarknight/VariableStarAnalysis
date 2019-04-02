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
import fit.astro.vsa.common.utilities.test.classification.GrabIrisData;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class PairWiseDistanceAnalysis {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(PairWiseDistanceAnalysis.class);
    
    public static void main(String[] args) throws IOException {

        GrabIrisData data = new GrabIrisData();

        Collection<RealVector> patterns = data.getSetOfPatterns().values();

        PairWiseDistances pairWiseDistances;
        RealMatrix distanceMatrix;
        RealMatrix distances;
        RealMatrix distancesParallel;

        long delta = 0;

        for (int i = 0; i < 101; i++) {
            //===================================== Parallel
            long startTimeParallel = System.nanoTime();

            pairWiseDistances
                    = new PairWiseDistances(new ArrayList<>(patterns));

            distanceMatrix
                    = pairWiseDistances.generateDistancesParallel(VectorDistanceType.EUCLIDEAN_DISTANCE);
            distancesParallel = distanceMatrix;

            long endTimeParallel = System.nanoTime();

            long durationParallel = (endTimeParallel - startTimeParallel);

            
//            LOGGER.info("Duration Parallel: " + durationParallel);
            // -------------------- Serial -------------------------------
            long startTime = System.nanoTime();

            pairWiseDistances
                    = new PairWiseDistances(new ArrayList<>(patterns));
            distanceMatrix
                    = pairWiseDistances.generateDistances(VectorDistanceType.EUCLIDEAN_DISTANCE);
            distances = distanceMatrix;

            long endTime = System.nanoTime();

            long durationSerial = (endTime - startTime);

//            LOGGER.info("Duration Serial: " + durationSerial);
            long deltak = durationParallel - durationSerial;

            RealMatrix deltaMatrix = distances.subtract(distancesParallel);
            double frobNorm = deltaMatrix.getFrobeniusNorm();

            if (i != 0) {
                delta += deltak / 100;
            }
//            LOGGER.info("Duration Delta: " + deltak + "  Frob Norm: " + frobNorm);

        }

        LOGGER.info("Duration Delta Mean: " + delta);

    }
}
