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
package fit.astro.vsa.common.utilities.math.support;

import java.util.Arrays;
import java.util.Map;
import org.apache.commons.math3.analysis.interpolation.SplineInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class InterpolationOperations {
    
    //Empty Constructor
    private InterpolationOperations(){
        
    }
    /**
     *
     * @param segmentPoints
     * @return
     */
    public static PolynomialSplineFunction GenerateSplineFromUnsortedMap( 
            Map<Double, Double> segmentPoints) {
        
        // Prep Spline A
        Double[] xArray = segmentPoints.keySet().toArray(new Double[segmentPoints.size()]);
        Double[] yArray = segmentPoints.values().toArray(new Double[segmentPoints.size()]);

        ArrayIndexComparator comparator = new ArrayIndexComparator(xArray);
        Integer[] indexesLon = comparator.createIndexArray();
        Arrays.sort(indexesLon, comparator);

        double[] xSorted = new double[xArray.length];
        double[] ySorted = new double[yArray.length];

        for (int i = 0; i < xArray.length; i++) {
            xSorted[i] = xArray[indexesLon[i]];
            ySorted[i] = yArray[indexesLon[i]];
        }

        SplineInterpolator spline = new SplineInterpolator();
        return spline.interpolate(xSorted, ySorted);
        
    }
}
