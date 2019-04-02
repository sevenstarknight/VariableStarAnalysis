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
import fit.astro.vsa.common.utilities.math.linearalgebra.VectorOperations;
import fit.astro.vsa.common.bindings.math.vector.SoftMaxFunction;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author SevenStarKnight
 */
public class SlottedTSGenerator {

    private final Real2DCurve waveformIn;
    private final double windowWidth;

    private double KERNELSPREAD = 0.01;


    /**
     *
     * @param waveformIn
     * @param windowWidth
     */
    public SlottedTSGenerator(Real2DCurve waveformIn,
            double windowWidth) {

        this.waveformIn = waveformIn;
        this.windowWidth = windowWidth;
    }

    /**
     *
     * @return
     */
    public List<List<Double>> generateSlottedTSGenerator() {

        // Initialize Waveform
        RealVector timeVector = waveformIn.getXVector();
        RealVector ampVector = waveformIn.getYVector();

        // ===============================================================
        // normalize time and amplitude
        RealVector x = VectorOperations.standardize(ampVector);
        RealVector t = timeVector.mapSubtract(timeVector.getMinValue());
        RealVector dt = VectorOperations.diffArray(t);

        List<Double> centers = getCenters(t, dt);
        return getSetsOfSequences(centers, x, t);
    }

    private List<Double> getCenters(RealVector t, RealVector dt) {
        //===================================================================
        // Find Slotting Centers Over Waveform
        List<Double> timeCenters = new ArrayList<>();
        double startIdx = 0.0;
        for (int idx = 0; idx < dt.getDimension(); idx++) {

            if (dt.getEntry(idx) >= 100) {

                double[] centers = VectorOperations.linearSpace(t.getEntry(idx),
                        startIdx, windowWidth);

                for (int jdx = 0; jdx < centers.length; jdx++) {
                    timeCenters.add(centers[jdx]);
                }

                startIdx = t.getEntry(idx + 1);

            }
        }

        if (timeCenters.isEmpty()) {
            double[] centers = VectorOperations.linearSpace(t.getMaxValue(),
                    startIdx, windowWidth);

            timeCenters.addAll(Arrays.asList(ArrayUtils.toObject(centers)));
        }

        return timeCenters;
    }

    private List<List<Double>> getSetsOfSequences(
            List<Double> timeCenters, RealVector x, RealVector t) {
        //===================================================================
        // Slot the Time Series data
        List<List<Double>> setOfSets = new ArrayList<>();
        List<Double> gaussianMeanSet = new ArrayList<>();

        for (Double e : timeCenters) {

            // Construct Gaussian Window 
            double upperTimeBound = e + windowWidth;
            double lowerTimeBound = e - windowWidth;

            // Linear Search
            List<Integer> inKernelIdx = new ArrayList<>();
            for (int idx = 0; idx < t.getDimension(); idx++) {
                if (t.getEntry(idx) <= upperTimeBound
                        && t.getEntry(idx) >= lowerTimeBound) {
                    inKernelIdx.add(idx);
                }
            }

            if (inKernelIdx.isEmpty()) {
                if (!gaussianMeanSet.isEmpty()) {
                    setOfSets.add(gaussianMeanSet);
                    gaussianMeanSet = new ArrayList<>();
                }
            } else {
                RealVector inKernelX = MatrixUtils.createRealVector(
                        new double[inKernelIdx.size()]);
                RealVector inKernelT = MatrixUtils.createRealVector(
                        new double[inKernelIdx.size()]);

                // pull using logic
                for (int idx = 0; idx < inKernelIdx.size(); idx++) {
                    inKernelX.setEntry(idx, x.getEntry(inKernelIdx.get(idx)));
                    inKernelT.setEntry(idx, t.getEntry(inKernelIdx.get(idx)));
                }

                RealVector weights = (inKernelT.map(
                        new SoftMaxFunction(e, KERNELSPREAD)));

                // estimate amplitude
                double mean = VectorOperations.summationOfElements(
                        weights.ebeMultiply(inKernelX))
                        / VectorOperations.summationOfElements(weights);

                // add to the list
                gaussianMeanSet.add(mean);
            }

        }

        setOfSets.add(gaussianMeanSet);

        return setOfSets;
    }
    //===================================================================

    /**
     * @param KERNELSPREAD the KERNELSPREAD to set
     */
    public void setKERNELSPREAD(double KERNELSPREAD) {
        this.KERNELSPREAD = KERNELSPREAD;
    }
}
