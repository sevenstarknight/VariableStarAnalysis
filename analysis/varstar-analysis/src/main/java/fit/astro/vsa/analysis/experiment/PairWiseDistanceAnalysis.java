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
package fit.astro.vsa.analysis.experiment;

import fit.astro.vsa.common.bindings.math.vector.VectorDistanceType;
import fit.astro.vsa.common.utilities.test.classification.GrabIrisData;
import fit.astro.vsa.utilities.ml.PairWiseDistances;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.random.MersenneTwister;
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

        int sizeArray = 100;
        List<RealVector> patterns = new ArrayList<>(sizeArray);
        NormalDistribution nd = new NormalDistribution(new MersenneTwister(42), 0.0, 1.0);
        for (int idx = 0; idx < sizeArray; idx++) {
            double[] sample = nd.sample(50);
            patterns.add(new ArrayRealVector(sample));
        }

        long delta = 0;

        PairWiseDistances pairWiseDistances
                = new PairWiseDistances(new ArrayList<>(patterns));

        for (int idx = 0; idx < sizeArray; idx++) {

            //===================================== Parallel
            long startTimeParallel = System.nanoTime();

            RealMatrix distanceMatrixA = pairWiseDistances
                    .generateDistancesParallel(VectorDistanceType.EUCLIDEAN_DISTANCE);

            long endTimeParallel = System.nanoTime();

            long durationParallel = (endTimeParallel - startTimeParallel);

            // -------------------- Serial -------------------------------
            long startTime = System.nanoTime();

            RealMatrix distanceMatrixB = pairWiseDistances.generateDistances(VectorDistanceType.EUCLIDEAN_DISTANCE);

            long endTime = System.nanoTime();

            long durationSerial = (endTime - startTime);

//            LOGGER.info("Duration Serial: " + durationSerial);
            long deltak = durationParallel - durationSerial;

            if (idx != 0) {
                delta += deltak / sizeArray;
            }
//            LOGGER.info("Duration Delta: " + deltak + "  Frob Norm: " + frobNorm);

        }

        LOGGER.info("Duration Delta Mean: " + delta);

    }
}
