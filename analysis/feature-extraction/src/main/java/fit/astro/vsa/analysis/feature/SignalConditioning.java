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
package fit.astro.vsa.analysis.feature;

import fit.astro.vsa.common.bindings.analysis.astro.VarStarDataset;
import fit.astro.vsa.common.bindings.math.Real2DCurve;
import fit.astro.vsa.common.utilities.io.dataset.ReadingInLINEARData;
import fit.astro.vsa.analysis.feature.smoothing.SuperSmoother;
import fit.astro.vsa.analysis.feature.smoothing.util.SuperSmootherProperties;
import fit.astro.vsa.common.bindings.math.vector.ModFunction;
import java.util.HashMap;
import java.util.Map;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class SignalConditioning {

    private SignalConditioning() {

    }

    /**
     *
     * @param waveform Input phased waveform
     * @return The Min-Max Waveform
     */
    public static Real2DCurve MinMaxNormalization(Real2DCurve waveform) {
        // ============================================
        //Min-Max Normalization

        RealVector yValues = waveform.getYVector();

        double maxValue = yValues.getMaxValue();
        double minValue = yValues.getMinValue();

        RealVector scaledY = (yValues.mapSubtract(minValue))
                .mapDivide(maxValue - minValue);

        //Min-Max
        return new Real2DCurve(waveform.getXVector(), scaledY);
    }

    /**
     *
     * @param phasedWaveform Input phased waveform (x [0,1])
     * @return The lag min zero waveform
     */
    public static Real2DCurve ShiftToMinZero(Real2DCurve phasedWaveform) {

        RealVector shiftedX = phasedWaveform.getXVector().mapSubtract(phasedWaveform.getXVector().getEntry(
                phasedWaveform.getYVector().getMinIndex()));

        for (int jdx = 0; jdx < shiftedX.getDimension(); jdx++) {
            if (shiftedX.getEntry(jdx) < 0.0) {
                shiftedX.setEntry(jdx, 1 + shiftedX.getEntry(jdx));
            }
        }

        return new Real2DCurve(shiftedX, phasedWaveform.getYVector());
    }

    /**
     *
     * @param dataset
     * @return
     */
    public static Map<Integer, Real2DCurve> generatePhasedData(VarStarDataset dataset) {

        SuperSmootherProperties props
                = new SuperSmootherProperties();

        Map<Integer, Real2DCurve> phasedData = new HashMap<>();

        for (Integer idx : dataset.getTimeD().keySet()) {

            Real2DCurve currentWaveform = dataset.getTimeD().get(idx);

            // ============================================================
            // Phase the data
            double truePeriod = Math.pow(10, dataset.getMultiViewSideData()
                    .getMapOfVectorPatterns().get(idx).get(ReadingInLINEARData.Views.TimeDomainVar.toString()).getEntry(0));

            RealVector time = currentWaveform.getXVector();

            RealVector phasedVector = (time.mapDivide(truePeriod)).map(new ModFunction(1.0));

            Real2DCurve phased2D = new Real2DCurve(phasedVector, currentWaveform.getYVector());

            // ============================================================
            // Sorted Phased Data With Window
            Real2DCurve sortedSeries = phased2D.getSortedSeries();

            Real2DCurve sortedSeriesMinus = new Real2DCurve(sortedSeries.getXVector().mapSubtract(1.0),
                    sortedSeries.getYVector());

            Real2DCurve sortedSeriesPlus = new Real2DCurve(sortedSeries.getXVector().mapAdd(1.0),
                    sortedSeries.getYVector());

            Real2DCurve sortedPlusMinus = new Real2DCurve(
                    sortedSeriesMinus.getXVector().append(sortedSeries.getXVector()).append(sortedSeriesPlus.getXVector()),
                    sortedSeriesMinus.getYVector().append(sortedSeries.getYVector()).append(sortedSeriesPlus.getYVector())
            );

            // ============================================================
            // Smoother
            SuperSmoother superSmoother = new SuperSmoother(sortedPlusMinus, props);

            RealVector ySmooth = superSmoother.execute().getSmo_n()
                    .getSubVector(sortedSeriesMinus.size(), sortedSeriesMinus.size());

            phasedData.put(idx, new Real2DCurve(sortedSeries.getXVector(), ySmooth));
        }

        return phasedData;
    }
}
