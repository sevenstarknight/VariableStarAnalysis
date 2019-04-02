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

import fit.astro.vsa.common.bindings.math.Real2DCurve;
import fit.astro.vsa.common.utilities.math.linearalgebra.MatrixOperations;
import fit.astro.vsa.common.utilities.math.linearalgebra.VectorOperations;
import java.util.Arrays;
import java.util.List;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import fit.astro.vsa.common.bindings.analysis.MatrixVariateTransform;

/**
 * Johnston, K. B., & Peter, A. M. (2017). Variable Star Signature
 * Classification using Slotted Symbolic Markov Modeling. New Astronomy, 50,
 * 1-11.
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class SSMMGenerator implements MatrixVariateTransform {

    private final List<Double> states;
    private final double windowWidth;

    /**
     * Make object with states
     *
     * @param states
     * @param windowWidth
     */
    public SSMMGenerator(List<Double> states, double windowWidth) {
        this.states = states;
        this.windowWidth = windowWidth;
    }

    /**
     * Make object with states
     *
     * @param resolution
     * @param windowWidth
     */
    public SSMMGenerator(double resolution, double windowWidth) {

        Double[] doubleArray = ArrayUtils.toObject(VectorOperations.linearSpace(2, -2, resolution));

        this.states = Arrays.asList(doubleArray);

        this.windowWidth = windowWidth;
    }

    /**
     *
     * @param waveformIn
     * @param scale
     * @return
     */
    public static double estimateWindowWidth(Real2DCurve waveformIn, double scale) {
        // Initialize Waveform
        RealVector timeVector = waveformIn.getXVector();
        RealVector t = timeVector.mapSubtract(timeVector.getMinValue());

        //===================================================================
        // Determine Resonable Slot Size
        RealVector dt = VectorOperations.diffArray(t);
        Percentile percentile = new Percentile();
        return percentile.evaluate(dt.toArray(), 50) * scale;
    }

    @Override
    public int[] getMatrixDimensions() {
        return new int[]{states.size(), states.size()};
    }

    /**
     *
     * @param waveformIn
     * @return
     */
    @Override
    public RealMatrix evaluate(Real2DCurve waveformIn) {
        //===================================================================
        SlottedTSGenerator sGenerator
                = new SlottedTSGenerator(waveformIn, windowWidth);

        List<List<Double>> setOfSequences
                = sGenerator.generateSlottedTSGenerator();
        //===================================================================

        MarkovModelGenerator mmg
                = new MarkovModelGenerator(setOfSequences, states);

        return mmg.getMc();
    }

    /**
     *
     * @param waveformIn
     * @return
     */
    public double[] getVectorizedMM(Real2DCurve waveformIn) {
        return MatrixOperations.unpackMatrix(evaluate(waveformIn));
    }

}
