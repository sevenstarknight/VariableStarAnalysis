/*
 * Copyright (C) 2019 kjohnston
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
package fit.astro.vsa.analysis.raw;

import fit.astro.vsa.analysis.feature.smoothing.SuperSmoother;
import fit.astro.vsa.analysis.feature.smoothing.util.SuperSmootherProperties;
import fit.astro.vsa.common.bindings.analysis.astro.VarStarDataset;
import fit.astro.vsa.common.bindings.math.Real2DCurve;
import fit.astro.vsa.common.bindings.math.vector.ModFunction;
import fit.astro.vsa.common.utilities.io.dataset.ReadingInLINEARData.Views;
import java.util.HashMap;
import java.util.Map;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author kjohnston
 */
public class SupportFunctionality {
    
    
    
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
                    .getMapOfVectorPatterns().get(idx).get(Views.TimeDomainVar.toString()).getEntry(0));

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
