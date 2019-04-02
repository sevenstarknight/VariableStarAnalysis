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
package fit.astro.vsa.analysis.ssmm.generators;

import fit.astro.vsa.common.utilities.math.linearalgebra.VectorOperations;
import java.util.List;
import java.util.NoSuchElementException;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author SevenStarKnight
 */
public class MarkovModelGenerator {

    private final RealMatrix mc;

    /**
     * Construct Markov Chain
     * <p>
     * @param setOfSequences
     * @param states
     */
    public MarkovModelGenerator(List<List<Double>> setOfSequences,
            List<Double> states) {

        // Transition State Frequency Model
        RealMatrix tsfm = MatrixUtils.createRealMatrix(
                states.size(), states.size());

        for (List<Double> seqX : setOfSequences) {
            RealMatrix tmpTSFM = constructTSFM(seqX, states);
            tsfm = tsfm.add(tmpTSFM);
        }

        mc = MatrixUtils.createRealMatrix(
                states.size(), states.size());

        for (int idx = 0; idx < tsfm.getRowDimension(); idx++) {
            RealVector rowVector = tsfm.getRowVector(idx);
            double sumRow = VectorOperations.summationOfElements(rowVector);

            if (sumRow != 0) {
                mc.setRowVector(idx, rowVector.mapDivide(sumRow));
            }
        }

    }

    private RealMatrix constructTSFM(List<Double> seqX, List<Double> states) {
        RealMatrix tsfm = MatrixUtils.createRealMatrix(
                states.size(), states.size());

        for (int idx = 1; idx < seqX.size(); idx++) {

            try {
                int startIdx = findIndex(seqX.get(idx - 1), states);
                int stopIdx = findIndex(seqX.get(idx), states);

                tsfm.setEntry(startIdx, stopIdx,
                        tsfm.getEntry(startIdx, stopIdx) + 1);
            } catch (NoSuchElementException me) {
                int startIdx = 0;
            }

        }

        return tsfm;
    }

    private int findIndex(double sample, List<Double> states) {

        if (states.get(states.size() - 1) <= sample) {
            return states.size() - 1;
        } else {
            Double currentState = states.stream()
                    .filter(x -> x > sample)
                    .findFirst().get();

            if (states.indexOf(currentState) == 0) {
                return 0;
            } else if (states.contains(currentState)) {
                return states.indexOf(currentState) - 1;
            } else {
                return 0;
            }
        }

    }

    public RealMatrix getMc() {
        return mc;
    }
}
